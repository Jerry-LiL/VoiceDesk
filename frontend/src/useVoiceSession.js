import { useCallback, useEffect, useRef, useState } from "react";

import { startCapture } from "./audio";

let playing = null;

function playWav(buffer) {
  if (playing) playing.pause();
  const url = URL.createObjectURL(new Blob([buffer], { type: "audio/wav" }));
  const audio = new Audio(url);
  playing = audio;
  const release = () => URL.revokeObjectURL(url);
  audio.onended = release;
  audio.play().catch(release);
}

export function useVoiceSession() {
  const wsRef = useRef(null);
  const captureRef = useRef(null);
  const holdingRef = useRef(false);
  const startedRef = useRef(false);
  const phaseRef = useRef("connecting");
  const [phase, setPhase] = useState("connecting");
  const [turns, setTurns] = useState([]);
  const [latency, setLatency] = useState(null);
  const [error, setError] = useState("");
  const [level, setLevel] = useState(0);
  const [sessionId, setSessionId] = useState("");

  const setPhaseSafe = useCallback((next) => {
    phaseRef.current = next;
    setPhase(next);
  }, []);

  const connect = useCallback(() => {
    const current = wsRef.current;
    if (
      current &&
      (current.readyState === WebSocket.OPEN || current.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${window.location.host}/ws`);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;
    setPhaseSafe("connecting");

    ws.onopen = () => {
      setError("");
      setPhaseSafe("ready");
    };
    ws.onclose = () => {
      // A replaced socket (a reconnect, or a dev reload) is not a failure.
      if (wsRef.current !== ws) return;
      holdingRef.current = false;
      startedRef.current = false;
      const lostLiveSession = phaseRef.current !== "connecting";
      setPhaseSafe("disconnected");
      if (lostLiveSession) setError("The connection dropped.");
    };
    ws.onerror = () => {};
    ws.onmessage = (event) => {
      if (typeof event.data !== "string") {
        playWav(event.data);
        return;
      }
      const msg = JSON.parse(event.data);
      if (msg.type === "ready") {
        setSessionId(msg.session_id || "");
        if (phaseRef.current === "connecting" || phaseRef.current === "disconnected") {
          setPhaseSafe("ready");
        }
      } else if (msg.type === "status" && msg.stage !== "listening") {
        setPhaseSafe(msg.stage);
      } else if (msg.type === "transcript") {
        setTurns((prev) => [...prev, { role: "user", text: msg.text }]);
      } else if (msg.type === "reply") {
        setTurns((prev) => [
          ...prev,
          {
            role: "assistant",
            text: msg.text,
            intent: msg.intent,
            fallback: msg.used_fallback,
          },
        ]);
      } else if (msg.type === "empty") {
        setError((prev) => prev || msg.message);
      } else if (msg.type === "error") {
        setError(msg.message);
      } else if (msg.type === "latency") {
        setLatency(msg);
      } else if (msg.type === "done") {
        if (!holdingRef.current && phaseRef.current !== "disconnected") {
          setPhaseSafe("ready");
        }
      } else if (msg.type === "reset") {
        setTurns([]);
        setLatency(null);
        setError("");
      }
    };
  }, [setPhaseSafe]);

  const begin = useCallback(
    async (event) => {
      event?.preventDefault?.();
      if (event?.currentTarget?.setPointerCapture && event.pointerId != null) {
        try {
          event.currentTarget.setPointerCapture(event.pointerId);
        } catch {
          // The pointer can already be gone. The keyboard path has no pointer.
        }
      }
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        connect();
        return;
      }
      if (phaseRef.current !== "ready" || holdingRef.current) return;
      setError("");
      holdingRef.current = true;
      setPhaseSafe("listening");
      ws.send(JSON.stringify({ type: "start", sample_rate: 16000 }));
      startedRef.current = true;
      try {
        const stop = await startCapture((pcm, rms) => {
          if (!holdingRef.current) return;
          if (ws.readyState === WebSocket.OPEN) ws.send(pcm);
          setLevel(rms);
        });
        if (!holdingRef.current) {
          stop();
          return;
        }
        captureRef.current = stop;
      } catch (err) {
        holdingRef.current = false;
        startedRef.current = false;
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "stop" }));
        }
        setPhaseSafe("ready");
        const blocked = err && err.name === "NotAllowedError";
        setError(
          blocked ? "Microphone permission was blocked." : "Could not open the microphone.",
        );
      }
    },
    [connect, setPhaseSafe],
  );

  const end = useCallback(() => {
    if (!holdingRef.current && !startedRef.current) return;
    holdingRef.current = false;
    const stop = captureRef.current;
    captureRef.current = null;
    if (stop) stop();
    setLevel(0);
    const ws = wsRef.current;
    if (startedRef.current && ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "stop" }));
      setPhaseSafe("transcribing");
    } else if (phaseRef.current === "listening") {
      setPhaseSafe("ready");
    }
    startedRef.current = false;
  }, [setPhaseSafe]);

  const sendText = useCallback((text) => {
    const cleaned = text.trim();
    const ws = wsRef.current;
    if (!cleaned || !ws || ws.readyState !== WebSocket.OPEN) return;
    if (phaseRef.current !== "ready") return;
    setError("");
    ws.send(JSON.stringify({ type: "text", text: cleaned }));
  }, []);

  const reset = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: "reset" }));
  }, []);

  useEffect(() => {
    connect();
    return () => {
      captureRef.current?.();
      wsRef.current?.close();
    };
  }, [connect]);

  useEffect(() => {
    function onKeyDown(event) {
      if (event.code !== "Space" || event.repeat) return;
      const tag = event.target && event.target.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      event.preventDefault();
      begin();
    }
    function onKeyUp(event) {
      if (event.code !== "Space") return;
      const tag = event.target && event.target.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      event.preventDefault();
      end();
    }
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [begin, end]);

  return { phase, turns, latency, error, level, sessionId, begin, end, sendText, reset };
}

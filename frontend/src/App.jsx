import { useEffect, useRef, useState } from "react";

import { useVoiceSession } from "./useVoiceSession";

const LABELS = {
  connecting: "Connecting",
  ready: "Hold to talk",
  listening: "Listening",
  transcribing: "Transcribing",
  thinking: "Thinking",
  speaking: "Speaking",
  disconnected: "Reconnect",
};

export default function App() {
  const voice = useVoiceSession();
  const [draft, setDraft] = useState("");
  const logRef = useRef(null);
  const busy = ["connecting", "transcribing", "thinking", "speaking"].includes(voice.phase);

  useEffect(() => {
    const node = logRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [voice.turns, voice.phase]);

  function onSubmit(event) {
    event.preventDefault();
    voice.sendText(draft);
    setDraft("");
  }

  return (
    <div className="page">
      <header className="top">
        <div>
          <p className="mark">VoiceDesk</p>
          <p className="sub">Push to talk</p>
        </div>
        <div className="status">
          <span className={voice.phase === "disconnected" ? "dot" : "dot on"} />
          <span>{voice.sessionId ? voice.sessionId : "offline"}</span>
        </div>
      </header>

      <ol className="log" ref={logRef}>
        {voice.turns.length === 0 && (
          <li className="placeholder">
            Hold the button and talk. You can also type a line if the microphone is unavailable.
          </li>
        )}
        {voice.turns.map((turn, index) => (
          <li key={index} className={turn.role === "user" ? "turn user" : "turn assistant"}>
            <span className="who">{turn.role === "user" ? "You" : "VoiceDesk"}</span>
            <p>{turn.text}</p>
            {turn.role === "assistant" && (
              <span className="intent">
                {turn.intent}
                {turn.fallback ? " · fallback" : ""}
              </span>
            )}
          </li>
        ))}
      </ol>

      {voice.error && <p className="error">{voice.error}</p>}
      {voice.latency && (
        <p className="latency">
          stt {voice.latency.stt_ms} ms · model {voice.latency.llm_ms} ms · speech{" "}
          {voice.latency.tts_ms} ms · total {voice.latency.total_ms} ms
        </p>
      )}

      <form className="composer" onSubmit={onSubmit}>
        <input
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder="Or type a line"
          aria-label="Type a line"
          disabled={voice.phase !== "ready"}
        />
        <button type="submit" disabled={voice.phase !== "ready" || !draft.trim()}>
          Send
        </button>
      </form>

      <div className="talk-row">
        <button
          type="button"
          className={voice.phase === "listening" ? "talk live" : busy ? "talk busy" : "talk"}
          onPointerDown={voice.begin}
          onPointerUp={voice.end}
          onPointerCancel={voice.end}
          aria-pressed={voice.phase === "listening"}
        >
          {LABELS[voice.phase] || "Hold to talk"}
        </button>
        <div className="meter" aria-hidden="true">
          <span style={{ width: `${Math.min(100, Math.round(voice.level * 280))}%` }} />
        </div>
        <div className="hints">
          <span>Hold the button or the space bar.</span>
          <button type="button" className="texty" onClick={voice.reset}>
            New conversation
          </button>
        </div>
      </div>
    </div>
  );
}

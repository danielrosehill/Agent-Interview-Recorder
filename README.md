# Agent Interview Recorder

Talk to a realtime voice agent that interviews you, and keep **your own side** of the conversation as speech
recognition training data: your raw microphone audio, your turns aligned to it, and a draft transcript of each
turn from the agent's own speech recognition.

Read speech is fluent and short. An interview gets long, spontaneous answers with real pauses, restarts and
run-ons, without you having to think of what to say.

## Status

| Stage | State |
|---|---|
| 1. Call page + local server (record, relay, log) | built, smoke-tested 2026-11-23 |
| 2. Split into per-turn takes (alignment) | planned, after the first 30-min test |
| 3. Second transcription + review page (only disagreements are reviewed) | planned |
| 4. Upload to a Hugging Face dataset | planned |

## Run

```bash
export GEMINI_API_KEY=...        # or: with-secrets uv run server.py
uv run server.py --prompt prompts/interviewer.md
# open http://localhost:8730, put headphones on, press Call
```

**Use headphones.** The mic is captured raw (echo cancellation, noise suppression and auto gain all off) because
it is training audio; through speakers the agent's voice would leak into your recording and into its own ears.

Options: `--model` (default `gemini-3.8-live`), `--voice` (default `Puck`), `--silence-ms` (default 1200: how long
a pause before the agent decides you have finished), `--out` (default `~/.local/share/agent-interview-recorder/sessions`).

## How it works

```
browser mic (48 kHz int16, raw) ──► server ──► <session>/mic.wav            (untouched, the training audio)
                                       └─resample 16 kHz──► Gemini Live API
browser speaker ◄── 24 kHz PCM ──── server ◄── agent audio, transcripts, turn events
                                       └──► <session>/events.jsonl, agent.wav
```

- The API key stays on the server; the browser only talks to `ws://localhost/ws`.
- Every event in `events.jsonl` carries `mic_sample`: how many mic samples had been written when it happened.
  That is the one clock for alignment. Kinds: `user_text` (the API's transcript of your turn, sent when your
  turn ends), `agent_text`, `agent_audio`, `interrupted` (you barged in), `turnComplete`,
  `client_playback_start` / `client_playback_end` (when the agent's voice actually played), `go_away`,
  `live_closed`, `usage`.
- Long calls: sliding-window context compression is on (without it Gemini Live audio sessions end at 15 min),
  and the server reconnects with the last session-resumption handle on `goAway` or a dropped connection
  (connections last about 10 min). The mic keeps recording through a reconnect.
- Turn-taking and barge-in are Gemini's server-side voice activity detection, set to low end-of-speech
  sensitivity so thinking pauses are not taken as the end of a turn.

## Findings

- 2026-11-23, smoke test (server driven by a script instead of a browser): with a real 2-minute dictation the
  agent did not cut in, and `user_text` came back near-verbatim (disfluencies and repetitions kept), stamped at
  each turn's end. A robotic espeak-ng voice was cut off after its first word and the rest never transcribed,
  so do not test with synthetic speech.
- Cost (Google pricing page, 2026-11-23): about $0.005/min audio in, $0.018/min audio out, so roughly
  $0.3 to 0.6 for 30 minutes.

## Session folder

`meta.json` (model, voice, prompt as sent), `mic.wav`, `agent.wav` (agent audio concatenated as received, no
gaps), `events.jsonl`.

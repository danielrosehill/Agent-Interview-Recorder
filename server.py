# /// script
# requires-python = ">=3.11"
# dependencies = ["aiohttp>=3.9", "websockets>=13", "numpy", "scipy"]
# ///
"""Agent Interview Recorder: local server.

The browser streams raw mic audio (48 kHz mono int16) to /ws. The server
 - writes it untouched to <session>/mic.wav (this is the training audio),
 - resamples a copy to 16 kHz and forwards it to the Gemini Live API,
 - relays the agent's 24 kHz audio back to the browser,
 - logs every event to <session>/events.jsonl stamped with `mic_sample`, the
   number of mic samples written at that moment (one clock for alignment).
It reconnects through session resumption on GoAway or a dropped connection.

Run: with-secrets uv run server.py --prompt PATH --topics PATH [--port 8730]
"""
import argparse, asyncio, base64, json, os, time, wave
from pathlib import Path

import numpy as np
import websockets
from aiohttp import web, WSMsgType
from scipy.signal import resample_poly

HERE = Path(__file__).parent
MIC_RATE = 48000
# Voices with a preview clip in static/voices/ (Gemini Live prebuilt voices, all checked on gemini-3.8-live 2026-11-23).
VOICES = sorted(p.stem for p in (HERE / "static" / "voices").glob("*.ogg"))
LIVE_URL = ("wss://generativelanguage.googleapis.com/ws/"
            "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=")


def setup_msg(args, prompt, voice, handle=None):
    s = {
        "model": f"models/{args.model}",
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
        "systemInstruction": {"parts": [{"text": prompt}]},
        "inputAudioTranscription": {},
        "outputAudioTranscription": {},
        "contextWindowCompression": {"slidingWindow": {}},
        "sessionResumption": {"handle": handle} if handle else {},
        "realtimeInputConfig": {"automaticActivityDetection": {
            # Thinking pauses are normal in an interview: be slow to decide he has finished.
            "endOfSpeechSensitivity": "END_SENSITIVITY_LOW",
            "silenceDurationMs": args.silence_ms,
        }},
    }
    return {"setup": s}


class Session:
    def __init__(self, args, prompt, browser, topic="", voice=None):
        self.args, self.browser = args, browser
        self.topic, self.voice = topic, voice or args.voice
        if topic:
            prompt += f"\n\nOpening topic for this call: {topic}. Start there; move to other topics when it runs dry."
        self.prompt = prompt
        stamp = time.strftime("%Y%m%dT%H%M%S")
        self.dir = Path(args.out).expanduser() / stamp
        self.dir.mkdir(parents=True, exist_ok=True)
        self.mic = wave.open(str(self.dir / "mic.wav"), "wb")
        self.mic.setnchannels(1); self.mic.setsampwidth(2); self.mic.setframerate(MIC_RATE)
        self.agent = wave.open(str(self.dir / "agent.wav"), "wb")  # agent audio, concatenated as received
        self.agent.setnchannels(1); self.agent.setsampwidth(2); self.agent.setframerate(24000)
        self.events = open(self.dir / "events.jsonl", "a")
        self.mic_samples = 0
        self.agent_samples = 0
        self.handle = None
        self.live = None
        self.closed = False
        (self.dir / "meta.json").write_text(json.dumps({
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "topic": topic, "model": args.model, "voice": self.voice,
            "mic_rate": MIC_RATE, "silence_ms": args.silence_ms, "prompt_file": str(args.prompt),
            "prompt": prompt}, indent=2))

    def log(self, kind, **kw):
        self.events.write(json.dumps({"t": round(time.time(), 3), "mic_sample": self.mic_samples,
                                      "kind": kind, **kw}, ensure_ascii=False) + "\n")
        self.events.flush()

    async def connect(self):
        self.live = await websockets.connect(LIVE_URL + os.environ["GEMINI_API_KEY"], max_size=None)
        await self.live.send(json.dumps(setup_msg(self.args, self.prompt, self.voice, self.handle)))
        reply = json.loads(await self.live.recv())
        self.log("live_connected", resumed=bool(self.handle), reply=list(reply))

    async def mic_in(self, pcm: bytes):
        self.mic.writeframes(pcm)
        self.mic_samples += len(pcm) // 2
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        y = np.clip(resample_poly(x, 1, 3), -32768, 32767).astype(np.int16)
        if self.live is not None:
            try:
                await self.live.send(json.dumps({"realtimeInput": {"audio": {
                    "data": base64.b64encode(y.tobytes()).decode(), "mimeType": "audio/pcm;rate=16000"}}}))
            except websockets.ConnectionClosed:
                pass  # the reader loop reconnects; audio meanwhile is still in mic.wav

    async def kick_off(self):
        # Let the agent open the conversation.
        await self.live.send(json.dumps({"realtimeInput": {"text": "(The call has connected. Greet me briefly and ask your first question.)"}}))

    async def live_loop(self):
        while not self.closed:
            try:
                async for raw in self.live:
                    await self.on_live(json.loads(raw))
            except websockets.ConnectionClosed as e:
                self.log("live_closed", code=e.code, reason=str(e.reason)[:200])
            if self.closed:
                break
            # Dropped or told to go away: resume with the last handle.
            self.live = None
            await self.browser.send_json({"type": "status", "text": "reconnecting"})
            for attempt in range(5):
                try:
                    await self.connect()
                    await self.browser.send_json({"type": "status", "text": "connected"})
                    break
                except Exception as e:  # noqa: BLE001
                    self.log("reconnect_failed", attempt=attempt, error=str(e)[:200])
                    await asyncio.sleep(1 + attempt)
            else:
                await self.browser.send_json({"type": "status", "text": "lost"})
                return

    async def on_live(self, m):
        if "sessionResumptionUpdate" in m:
            u = m["sessionResumptionUpdate"]
            if u.get("resumable") and u.get("newHandle"):
                self.handle = u["newHandle"]
            return
        if "goAway" in m:
            self.log("go_away", detail=m["goAway"])
            await self.live.close()
            return
        if "usageMetadata" in m:
            self.log("usage", usage=m["usageMetadata"])
        sc = m.get("serverContent")
        if not sc:
            return
        for p in sc.get("modelTurn", {}).get("parts", []):
            if "inlineData" in p:
                pcm = base64.b64decode(p["inlineData"]["data"])
                self.log("agent_audio", agent_sample=self.agent_samples, n=len(pcm) // 2)
                self.agent.writeframes(pcm)
                self.agent_samples += len(pcm) // 2
                await self.browser.send_bytes(pcm)
        if "inputTranscription" in sc:
            txt = sc["inputTranscription"].get("text", "")
            self.log("user_text", text=txt)
            await self.browser.send_json({"type": "user_text", "text": txt})
        if "outputTranscription" in sc:
            txt = sc["outputTranscription"].get("text", "")
            self.log("agent_text", text=txt)
            await self.browser.send_json({"type": "agent_text", "text": txt})
        for flag in ("interrupted", "generationComplete", "turnComplete"):
            if sc.get(flag):
                self.log(flag)
                await self.browser.send_json({"type": flag})

    async def close(self):
        self.closed = True
        self.log("hangup")
        if self.live is not None:
            await self.live.close()
        self.mic.close(); self.agent.close(); self.events.close()
        return self.mic_samples / MIC_RATE


async def ws_handler(request):
    args, prompt = request.app["args"], request.app["prompt"]
    browser = web.WebSocketResponse(max_msg_size=0)
    await browser.prepare(request)
    first = await browser.receive_json()  # {"type": "start", "topic": ..., "voice": ...}
    voice = first.get("voice") if first.get("voice") in VOICES else None
    s = Session(args, prompt, browser, topic=str(first.get("topic", ""))[:200], voice=voice)
    try:
        await s.connect()
    except Exception as e:  # noqa: BLE001
        await browser.send_json({"type": "status", "text": f"failed: {e}"})
        await browser.close()
        return browser
    await browser.send_json({"type": "session", "dir": str(s.dir)})
    reader = asyncio.create_task(s.live_loop())
    await s.kick_off()
    async for msg in browser:
        if msg.type == WSMsgType.BINARY:
            await s.mic_in(msg.data)
        elif msg.type == WSMsgType.TEXT:
            m = json.loads(msg.data)
            if m.get("type") == "hangup":
                break
            s.log("client_" + m.get("type", "event"), **{k: v for k, v in m.items() if k != "type"})
    secs = await s.close()
    reader.cancel()
    print(f"saved {secs/60:.1f} min to {s.dir}")
    return browser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=str(HERE / "prompts" / "interviewer.md"))
    ap.add_argument("--topics", default=str(HERE / "prompts" / "topics.json"))
    ap.add_argument("--model", default="gemini-3.8-live")
    ap.add_argument("--voice", default="Sulafat")
    ap.add_argument("--silence-ms", type=int, default=1200)
    ap.add_argument("--out", default="~/.local/share/agent-interview-recorder/sessions")
    ap.add_argument("--port", type=int, default=8730)
    args = ap.parse_args()
    app = web.Application()
    app["args"], app["prompt"] = args, Path(args.prompt).expanduser().read_text()
    app["topics"] = json.loads(Path(args.topics).expanduser().read_text())
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/config", lambda r: web.json_response({"topics": r.app["topics"], "voices": VOICES, "voice": args.voice}))
    app.router.add_get("/", lambda r: web.FileResponse(HERE / "static" / "index.html"))
    app.router.add_static("/static", HERE / "static")
    print(f"open http://localhost:{args.port}  (prompt: {args.prompt})")
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()

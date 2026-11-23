# Record a preview line in each named Gemini Live voice: with-secrets uv run --with websockets python tools/voice_samples.py Sulafat Charon ...  (writes /tmp/gl/voices/<Voice>.wav)
import asyncio,json,os,base64,wave,sys,websockets
URL="wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key="+os.environ["GEMINI_API_KEY"]
LINE="So you moved from Ireland to Jerusalem. What was the first thing that really surprised you about daily life here?"
async def one(v):
  async with websockets.connect(URL,max_size=None) as ws:
    await ws.send(json.dumps({"setup":{"model":"models/gemini-3.8-live","generationConfig":{"responseModalities":["AUDIO"],"speechConfig":{"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":v}}}},
      "systemInstruction":{"parts":[{"text":"You are a calm, unhurried, thoughtful interviewer. Speak warmly and quietly, never gushing. Say exactly the sentence you are given, nothing else."}]}}}))
    r=json.loads(await ws.recv())
    await ws.send(json.dumps({"realtimeInput":{"text":"Say: "+LINE}}))
    pcm=b""
    while True:
      m=json.loads(await asyncio.wait_for(ws.recv(),40));sc=m.get("serverContent",{})
      for p in sc.get("modelTurn",{}).get("parts",[]):
        if "inlineData" in p: pcm+=base64.b64decode(p["inlineData"]["data"])
      if sc.get("turnComplete"): break
  out=wave.open(f"/tmp/gl/voices/{v}.wav","wb");out.setnchannels(1);out.setsampwidth(2);out.setframerate(24000);out.writeframes(pcm);out.close()
  print(v,round(len(pcm)/48000,1),"s")
async def main():
  os.makedirs("/tmp/gl/voices",exist_ok=True)
  for r in await asyncio.gather(*[one(v) for v in sys.argv[1:]],return_exceptions=True):
    if isinstance(r,Exception): print("ERR",r)
asyncio.run(main())

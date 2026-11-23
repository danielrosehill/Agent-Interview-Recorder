"""Drive the local server without a browser: streams a real speech WAV (not synthetic: espeak gets cut off) as the mic.
Usage: uv run --with aiohttp --with numpy --with scipy python tests/smoke_client.py speech.wav  (server on port 8731)
"""
import asyncio, json, wave, numpy as np, aiohttp
from scipy.signal import resample_poly
import sys
w=wave.open(sys.argv[1] if len(sys.argv)>1 else 'say.wav'); sr=w.getframerate(); x=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float32)
y=resample_poly(x,48000,sr).astype(np.int16)
sil=np.zeros(48000*3,dtype=np.int16)
stream=np.concatenate([sil,sil,sil,y,sil,sil,sil,sil])  # 9 s silence (agent greets), speech, 12 s silence
async def main():
  async with aiohttp.ClientSession() as s, s.ws_connect('http://127.0.0.1:8731/ws') as ws:
    await ws.send_str(json.dumps({"type":"start","topic":"Shopping at the makolet","voice":"Sulafat"}))
    async def rx():
      agent=0
      async for m in ws:
        if m.type==aiohttp.WSMsgType.BINARY: agent+=len(m.data)//2
        else:
          d=json.loads(m.data)
          if d['type'] in('user_text','agent_text'): print(d['type'],repr(d['text']))
          elif d['type']!='generationComplete': print(d)
      print('agent samples',agent)
    t=asyncio.create_task(rx())
    for i in range(0,len(stream),4800):
      await ws.send_bytes(stream[i:i+4800].tobytes()); await asyncio.sleep(0.1)
    await ws.send_str(json.dumps({"type":"hangup"})); await asyncio.sleep(1); t.cancel()
asyncio.run(main())

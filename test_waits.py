from mjai_bot.mortal.libriichi.state import PlayerState
import json

p = PlayerState(0)
# Feed events to get to an iishanten state
events = [
  {"type":"start_game","names":["0","1","2","3"],"id":0},
  {"type":"start_kyoku","bakaze":"S","dora_marker":"1p","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"]]}
]
for e in events:
    p.update(json.dumps(e, separators=(",", ":")))

print("shanten:", p.shanten)
print("waits:", p.waits)

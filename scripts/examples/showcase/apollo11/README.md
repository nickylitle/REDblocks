# Apollo 11 in Simblocks

Open **apollo11.html** in any browser (no internet needed). It shows both canvases against NASA's records: every curve
is the sim, every dot is NASA's.

| File | What it is |
|---|---|
| `apollo11_1_ascent.yaml` | Saturn V AS-506 from first motion to Earth parking orbit: F-1 and J-2 engines from propellant chemistry, staging, drag, the tilt program, a gravity turn and guidance to orbit |
| `apollo11_2_landing.yaml` | Lunar module Eagle from powered descent initiation to touchdown: the descent engine (Aerozine 50 / N2O4), Apollo's quadratic guidance law in P63 and P64, throttle recovery, and P66 flown from the crew's callouts |
| `nasa_reference.json` | The measured and planned values the canvases are checked against, each with its source |
| `build_page.py` | Runs the canvases and writes `apollo11.html` |
| `apollo11.html` | The results page |

## Run it

```
simblocks run examples/showcase/apollo11/apollo11_1_ascent.yaml
simblocks run examples/showcase/apollo11/apollo11_2_landing.yaml
simblocks set examples/showcase/apollo11/apollo11_2_landing.yaml crew=0     # the automatic landing NASA planned
python examples/showcase/apollo11/build_page.py                             # rebuild the page
simblocks app                                                               # open a canvas in the node editor
```

## What is given and what is simulated

Given: masses, engine ratings, thrust levels and event times from the flight reports; NASA's premission state at
descent ignition and its published high gate; the descent rates the crew called out below 400 ft. One number is
tuned: the Saturn V pitch-over rate. Simulated: every engine's chamber temperature and Isp from chemical equilibrium,
and the whole trajectory of both flights, every 20 ms, from thrust, drag and gravity, with steering and throttle
from guidance laws.

## Where it misses

The page lists every miss. The largest: the ascent's max dynamic pressure is 7 % low (standard atmosphere and a
generic drag curve); the descent's throttle recovery comes about 10 s late and the approach runs about 20 s ahead
(Apollo 11's own guidance targets were loaded before flight and are not in these sources); the lander uses 3.4 %
less propellant than Eagle did.

Left out on purpose: the translunar injection burn, the coast to the Moon and lunar orbit insertion. The showcase
covers the two powered flights: the climb out of Earth's atmosphere and the guided landing.

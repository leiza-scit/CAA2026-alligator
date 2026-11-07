/* Written by py/build_dating.py from output/horizon_reference_profiles.csv
   and output/arretine_sites_minigraph.ttl. */
window.DMReference = {
 "source": "output/horizon_reference_profiles.csv",
 "intervalSource": "output/arretine_sites_minigraph.ttl",
 "labels": [
  "BordOblique",
  "Ia",
  "Ib",
  "Ic",
  "II"
 ],
 "countsSha256": "342fd585f20733c3d56803e0672c8115afe2f4550458b1cfb2b821f6b1e5fe1c",
 "allen": [
  {
   "from": 1,
   "rel": "time:intervalStarts",
   "to": 2
  },
  {
   "from": 1,
   "rel": "time:intervalMeets",
   "to": 3
  },
  {
   "from": 3,
   "rel": "time:intervalDuring",
   "to": 2
  },
  {
   "from": 3,
   "rel": "time:intervalOverlaps",
   "to": 4
  },
  {
   "from": 4,
   "rel": "time:intervalFinishes",
   "to": 2
  }
 ],
 "horizons": [
  {
   "horizon": 1,
   "n_total": 482,
   "counts": [
    6,
    49,
    153,
    244,
    30
   ],
   "iri": "http://leiza-scit.github.io/CAA2026-alligator/collections/horizon_1",
   "label": "Horizon 1 (15 BCE – 7 BCE)",
   "begin": -15,
   "end": -7
  },
  {
   "horizon": 2,
   "n_total": 2706,
   "counts": [
    4,
    115,
    661,
    965,
    961
   ],
   "iri": "http://leiza-scit.github.io/CAA2026-alligator/collections/horizon_2",
   "label": "Horizon 2 (15 BCE – 13 CE)",
   "begin": -15,
   "end": 13
  },
  {
   "horizon": 3,
   "n_total": 2132,
   "counts": [
    0,
    8,
    216,
    684,
    1224
   ],
   "iri": "http://leiza-scit.github.io/CAA2026-alligator/collections/horizon_3",
   "label": "Horizon 3 (7 BCE – 9 CE)",
   "begin": -7,
   "end": 9
  },
  {
   "horizon": 4,
   "n_total": 476,
   "counts": [
    0,
    1,
    32,
    116,
    327
   ],
   "iri": "http://leiza-scit.github.io/CAA2026-alligator/collections/horizon_4",
   "label": "Horizon 4 (8 CE – 13 CE)",
   "begin": 8,
   "end": 13
  },
  {
   "horizon": 5,
   "n_total": 1334,
   "counts": [
    0,
    0,
    11,
    92,
    1231
   ],
   "iri": "http://leiza-scit.github.io/CAA2026-alligator/collections/horizon_5",
   "label": "Horizon 5 (16 CE – 28 CE)",
   "begin": 16,
   "end": 28
  }
 ]
};

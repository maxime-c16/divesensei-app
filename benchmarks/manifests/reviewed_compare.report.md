# DiveSensei Detector Validation

Manifest: `/home/mcauchy/divesensei-app/benchmarks/manifests/reviewed_compare.json`

## Detector Summary

### `audio_v1_heuristic`

- Existing cases: 42
- Pass rate: 0.7619047619047619
- Precision: n/a
- Recall: n/a
- Mean runtime seconds: 0.276

### `audio_v2_pcen_classifier`

- Existing cases: 42
- Pass rate: 0.7857142857142857
- Precision: n/a
- Recall: n/a
- Mean runtime seconds: 0.361

## Cases

### PASS - `IMG_2478.MOV` [audio_v1_heuristic]

- Detected events: `1`
- Runtime seconds: `0.29775285720825195`
- Predicted timestamps: `[6.416]`
- Notes: Reviewed positive: diver visible above water near platform.

### PASS - `IMG_2496.MOV` [audio_v1_heuristic]

- Detected events: `1`
- Runtime seconds: `0.1970067024230957`
- Predicted timestamps: `[2.544]`
- Notes: Reviewed positive: diver inverted above pool.

### PASS - `IMG_8148.MOV` [audio_v1_heuristic]

- Detected events: `1`
- Runtime seconds: `0.27088212966918945`
- Predicted timestamps: `[15.408]`
- Notes: Reviewed positive: pool dive clip.

### PASS - `IMG_8150.MOV` [audio_v1_heuristic]

- Detected events: `1`
- Runtime seconds: `0.28513240814208984`
- Predicted timestamps: `[19.456]`
- Notes: Reviewed positive: pool dive clip.

### PASS - `IMG_8151.MOV` [audio_v1_heuristic]

- Detected events: `1`
- Runtime seconds: `0.3690018653869629`
- Predicted timestamps: `[23.792]`
- Notes: Reviewed positive: pool dive clip.

### FAIL - `IMG_8152.MOV` [audio_v1_heuristic]

- Detected events: `4`
- Runtime seconds: `0.5712239742279053`
- Predicted timestamps: `[0.0, 5.136, 23.824, 30.16]`
- Notes: Reviewed positive: pool dive clip.

### PASS - `IMG_8154.MOV` [audio_v1_heuristic]

- Detected events: `3`
- Runtime seconds: `0.3275129795074463`
- Predicted timestamps: `[0.848, 18.96, 23.312]`
- Notes: Reviewed positive: athlete on springboard/platform over pool.

### PASS - `IMG_8155.MOV` [audio_v1_heuristic]

- Detected events: `1`
- Runtime seconds: `0.26792407035827637`
- Predicted timestamps: `[3.808]`
- Notes: Reviewed positive: splash/pool dive clip.

### PASS - `IMG_8156.MOV` [audio_v1_heuristic]

- Detected events: `1`
- Runtime seconds: `0.25551342964172363`
- Predicted timestamps: `[6.208]`
- Notes: Reviewed positive: athlete on board above pool.

### FAIL - `IMG_8162.MOV` [audio_v1_heuristic]

- Detected events: `0`
- Runtime seconds: `0.17229413986206055`
- Predicted timestamps: `[]`
- Notes: Reviewed positive: airborne diver over pool.

### PASS - `IMG_8164.MOV` [audio_v1_heuristic]

- Detected events: `2`
- Runtime seconds: `0.20709466934204102`
- Predicted timestamps: `[4.944, 9.248]`
- Notes: Reviewed positive: athletes on board over pool.

### PASS - `IMG_8166.MOV` [audio_v1_heuristic]

- Detected events: `1`
- Runtime seconds: `0.1737349033355713`
- Predicted timestamps: `[3.888]`
- Notes: Reviewed positive: airborne diver.

### PASS - `IMG_8170.MOV` [audio_v1_heuristic]

- Detected events: `2`
- Runtime seconds: `0.21547985076904297`
- Predicted timestamps: `[0.784, 11.952]`
- Notes: Reviewed positive: pool dive clip.

### PASS - `IMG_8175.MOV` [audio_v1_heuristic]

- Detected events: `1`
- Runtime seconds: `0.18931174278259277`
- Predicted timestamps: `[0.0]`
- Notes: Reviewed positive: diver above platform.

### PASS - `IMG_8181.MOV` [audio_v1_heuristic]

- Detected events: `1`
- Runtime seconds: `0.20093059539794922`
- Predicted timestamps: `[6.864]`
- Notes: Reviewed positive: diver in motion over pool.

### PASS - `IMG_8182.MOV` [audio_v1_heuristic]

- Detected events: `1`
- Runtime seconds: `0.17551779747009277`
- Predicted timestamps: `[5.536]`
- Notes: Reviewed positive: airborne diver over pool.

### PASS - `IMG_8183.MOV` [audio_v1_heuristic]

- Detected events: `2`
- Runtime seconds: `0.21025943756103516`
- Predicted timestamps: `[0.0, 8.064]`
- Notes: Reviewed positive: pool dive clip.

### PASS - `IMG_8184.MOV` [audio_v1_heuristic]

- Detected events: `1`
- Runtime seconds: `0.21098804473876953`
- Predicted timestamps: `[14.336]`
- Notes: Reviewed positive: airborne diver over pool.

### PASS - `IMG_8185.MOV` [audio_v1_heuristic]

- Detected events: `2`
- Runtime seconds: `0.24928569793701172`
- Predicted timestamps: `[6.416, 17.328]`
- Notes: Reviewed positive: athlete on board above pool.

### PASS - `IMG_8186.MOV` [audio_v1_heuristic]

- Detected events: `1`
- Runtime seconds: `0.19742846488952637`
- Predicted timestamps: `[9.008]`
- Notes: Reviewed positive: diving platform/pool clip.

### PASS - `IMG_8187.MOV` [audio_v1_heuristic]

- Detected events: `1`
- Runtime seconds: `0.23785829544067383`
- Predicted timestamps: `[1.536]`
- Notes: Reviewed positive: platform dive clip.

### PASS - `IMG_8190.MOV` [audio_v1_heuristic]

- Detected events: `1`
- Runtime seconds: `0.2082500457763672`
- Predicted timestamps: `[11.36]`
- Notes: Reviewed positive: diver hanging above pool.

### PASS - `IMG_8201.MOV` [audio_v1_heuristic]

- Detected events: `1`
- Runtime seconds: `0.20022988319396973`
- Predicted timestamps: `[0.352]`
- Notes: Reviewed positive: diver on platform above pool.

### PASS - `IMG_8206.MOV` [audio_v1_heuristic]

- Detected events: `1`
- Runtime seconds: `0.20400691032409668`
- Predicted timestamps: `[12.016]`
- Notes: Reviewed positive: diver on platform above pool.

### PASS - `IMG_8208.MOV` [audio_v1_heuristic]

- Detected events: `1`
- Runtime seconds: `0.15052270889282227`
- Predicted timestamps: `[0.0]`
- Notes: Reviewed positive: water-entry frame with divers descending into pool.

### PASS - `IMG_8209.MOV` [audio_v1_heuristic]

- Detected events: `1`
- Runtime seconds: `0.2645864486694336`
- Predicted timestamps: `[18.928]`
- Notes: Reviewed positive: diver poised on springboard over pool.

### PASS - `IMG_8210.MOV` [audio_v1_heuristic]

- Detected events: `3`
- Runtime seconds: `0.2810075283050537`
- Predicted timestamps: `[4.192, 11.952, 22.736]`
- Notes: Reviewed positive: athletes poised on springboard over pool.

### PASS - `IMG_8221.MOV` [audio_v1_heuristic]

- Detected events: `1`
- Runtime seconds: `0.23031020164489746`
- Predicted timestamps: `[12.432]`
- Notes: Reviewed positive: diver in takeoff position above pool.

### FAIL - `IMG_8222.MOV` [audio_v1_heuristic]

- Detected events: `4`
- Runtime seconds: `0.2673811912536621`
- Predicted timestamps: `[0.64, 7.232, 14.32, 24.0]`
- Notes: Reviewed positive: diver in takeoff position above pool.

### PASS - `IMG_8231.MOV` [audio_v1_heuristic]

- Detected events: `2`
- Runtime seconds: `0.2504405975341797`
- Predicted timestamps: `[2.96, 8.768]`
- Notes: Reviewed positive: diver airborne above pool.

### PASS - `IMG_8088.MOV` [audio_v1_heuristic]

- Detected events: `0`
- Runtime seconds: `0.18598341941833496`
- Predicted timestamps: `[]`
- Notes: Reviewed negative: indoor social clip.

### PASS - `IMG_8089.MOV` [audio_v1_heuristic]

- Detected events: `0`
- Runtime seconds: `0.18895626068115234`
- Predicted timestamps: `[]`
- Notes: Reviewed negative: indoor selfie/retail clip.

### PASS - `IMG_8090.MOV` [audio_v1_heuristic]

- Detected events: `0`
- Runtime seconds: `0.8586783409118652`
- Predicted timestamps: `[]`
- Notes: Reviewed negative: hotel/interior clip.

### FAIL - `IMG_8098.MOV` [audio_v1_heuristic]

- Detected events: `3`
- Runtime seconds: `0.6170456409454346`
- Predicted timestamps: `[5.712, 40.576, 72.656]`
- Notes: Reviewed negative: bathroom/interior clip.

### FAIL - `IMG_8099.MOV` [audio_v1_heuristic]

- Detected events: `2`
- Runtime seconds: `0.3395726680755615`
- Predicted timestamps: `[2.8, 7.12]`
- Notes: Reviewed negative: hotel/interior clip.

### FAIL - `IMG_8105.MOV` [audio_v1_heuristic]

- Detected events: `5`
- Runtime seconds: `0.5361607074737549`
- Predicted timestamps: `[2.384, 24.544, 42.608, 55.6, 61.536]`
- Notes: Reviewed negative: dryland training hall, no splash event.

### FAIL - `IMG_8212.MOV` [audio_v1_heuristic]

- Detected events: `1`
- Runtime seconds: `0.23136520385742188`
- Predicted timestamps: `[6.416]`
- Notes: Reviewed negative: indoor selfie clip.

### FAIL - `IMG_8214.MOV` [audio_v1_heuristic]

- Detected events: `2`
- Runtime seconds: `0.25282740592956543`
- Predicted timestamps: `[1.632, 7.472]`
- Notes: Reviewed negative: abstract close-up, no pool or dive.

### PASS - `IMG_8215.MOV` [audio_v1_heuristic]

- Detected events: `0`
- Runtime seconds: `0.3088963031768799`
- Predicted timestamps: `[]`
- Notes: Reviewed negative: elevator/interior clip.

### FAIL - `IMG_8216.MOV` [audio_v1_heuristic]

- Detected events: `1`
- Runtime seconds: `0.29204416275024414`
- Predicted timestamps: `[17.568]`
- Notes: Reviewed negative: koi pond/outdoor water clip, not diving.

### FAIL - `IMG_8217.MOV` [audio_v1_heuristic]

- Detected events: `1`
- Runtime seconds: `0.26023054122924805`
- Predicted timestamps: `[10.064]`
- Notes: Reviewed negative: phone/selfie clip.

### PASS - `IMG_8227.MOV` [audio_v1_heuristic]

- Detected events: `0`
- Runtime seconds: `0.1629047393798828`
- Predicted timestamps: `[]`
- Notes: Reviewed negative: selfie/ground close-up clip.

### FAIL - `IMG_2478.MOV` [audio_v2_pcen_classifier]

- Detected events: `0`
- Runtime seconds: `0.2544679641723633`
- Predicted timestamps: `[]`
- Notes: Reviewed positive: diver visible above water near platform.

### PASS - `IMG_2496.MOV` [audio_v2_pcen_classifier]

- Detected events: `1`
- Runtime seconds: `0.20533037185668945`
- Predicted timestamps: `[2.592]`
- Notes: Reviewed positive: diver inverted above pool.

### FAIL - `IMG_8148.MOV` [audio_v2_pcen_classifier]

- Detected events: `0`
- Runtime seconds: `0.34089183807373047`
- Predicted timestamps: `[]`
- Notes: Reviewed positive: pool dive clip.

### PASS - `IMG_8150.MOV` [audio_v2_pcen_classifier]

- Detected events: `1`
- Runtime seconds: `0.34377193450927734`
- Predicted timestamps: `[15.472]`
- Notes: Reviewed positive: pool dive clip.

### FAIL - `IMG_8151.MOV` [audio_v2_pcen_classifier]

- Detected events: `0`
- Runtime seconds: `0.5343348979949951`
- Predicted timestamps: `[]`
- Notes: Reviewed positive: pool dive clip.

### PASS - `IMG_8152.MOV` [audio_v2_pcen_classifier]

- Detected events: `3`
- Runtime seconds: `0.5737063884735107`
- Predicted timestamps: `[0.0, 5.136, 23.824]`
- Notes: Reviewed positive: pool dive clip.

### PASS - `IMG_8154.MOV` [audio_v2_pcen_classifier]

- Detected events: `2`
- Runtime seconds: `0.4896864891052246`
- Predicted timestamps: `[0.848, 18.96]`
- Notes: Reviewed positive: athlete on springboard/platform over pool.

### PASS - `IMG_8155.MOV` [audio_v2_pcen_classifier]

- Detected events: `1`
- Runtime seconds: `0.29844212532043457`
- Predicted timestamps: `[4.464]`
- Notes: Reviewed positive: splash/pool dive clip.

### PASS - `IMG_8156.MOV` [audio_v2_pcen_classifier]

- Detected events: `1`
- Runtime seconds: `0.29763293266296387`
- Predicted timestamps: `[6.208]`
- Notes: Reviewed positive: athlete on board above pool.

### FAIL - `IMG_8162.MOV` [audio_v2_pcen_classifier]

- Detected events: `0`
- Runtime seconds: `0.23129725456237793`
- Predicted timestamps: `[]`
- Notes: Reviewed positive: airborne diver over pool.

### FAIL - `IMG_8164.MOV` [audio_v2_pcen_classifier]

- Detected events: `0`
- Runtime seconds: `0.3343467712402344`
- Predicted timestamps: `[]`
- Notes: Reviewed positive: athletes on board over pool.

### PASS - `IMG_8166.MOV` [audio_v2_pcen_classifier]

- Detected events: `1`
- Runtime seconds: `0.24219012260437012`
- Predicted timestamps: `[3.968]`
- Notes: Reviewed positive: airborne diver.

### PASS - `IMG_8170.MOV` [audio_v2_pcen_classifier]

- Detected events: `1`
- Runtime seconds: `0.3011162281036377`
- Predicted timestamps: `[0.784]`
- Notes: Reviewed positive: pool dive clip.

### PASS - `IMG_8175.MOV` [audio_v2_pcen_classifier]

- Detected events: `1`
- Runtime seconds: `0.259519100189209`
- Predicted timestamps: `[0.0]`
- Notes: Reviewed positive: diver above platform.

### FAIL - `IMG_8181.MOV` [audio_v2_pcen_classifier]

- Detected events: `0`
- Runtime seconds: `0.26201415061950684`
- Predicted timestamps: `[]`
- Notes: Reviewed positive: diver in motion over pool.

### PASS - `IMG_8182.MOV` [audio_v2_pcen_classifier]

- Detected events: `1`
- Runtime seconds: `0.21327543258666992`
- Predicted timestamps: `[5.536]`
- Notes: Reviewed positive: airborne diver over pool.

### PASS - `IMG_8183.MOV` [audio_v2_pcen_classifier]

- Detected events: `2`
- Runtime seconds: `0.25687694549560547`
- Predicted timestamps: `[0.0, 8.064]`
- Notes: Reviewed positive: pool dive clip.

### PASS - `IMG_8184.MOV` [audio_v2_pcen_classifier]

- Detected events: `1`
- Runtime seconds: `0.2764315605163574`
- Predicted timestamps: `[14.336]`
- Notes: Reviewed positive: airborne diver over pool.

### PASS - `IMG_8185.MOV` [audio_v2_pcen_classifier]

- Detected events: `3`
- Runtime seconds: `0.3364865779876709`
- Predicted timestamps: `[3.104, 6.416, 17.328]`
- Notes: Reviewed positive: athlete on board above pool.

### FAIL - `IMG_8186.MOV` [audio_v2_pcen_classifier]

- Detected events: `0`
- Runtime seconds: `0.2919316291809082`
- Predicted timestamps: `[]`
- Notes: Reviewed positive: diving platform/pool clip.

### PASS - `IMG_8187.MOV` [audio_v2_pcen_classifier]

- Detected events: `2`
- Runtime seconds: `0.324725866317749`
- Predicted timestamps: `[1.536, 11.12]`
- Notes: Reviewed positive: platform dive clip.

### PASS - `IMG_8190.MOV` [audio_v2_pcen_classifier]

- Detected events: `1`
- Runtime seconds: `0.31545042991638184`
- Predicted timestamps: `[0.96]`
- Notes: Reviewed positive: diver hanging above pool.

### FAIL - `IMG_8201.MOV` [audio_v2_pcen_classifier]

- Detected events: `0`
- Runtime seconds: `0.26058197021484375`
- Predicted timestamps: `[]`
- Notes: Reviewed positive: diver on platform above pool.

### FAIL - `IMG_8206.MOV` [audio_v2_pcen_classifier]

- Detected events: `0`
- Runtime seconds: `0.24733400344848633`
- Predicted timestamps: `[]`
- Notes: Reviewed positive: diver on platform above pool.

### PASS - `IMG_8208.MOV` [audio_v2_pcen_classifier]

- Detected events: `1`
- Runtime seconds: `0.16228580474853516`
- Predicted timestamps: `[0.0]`
- Notes: Reviewed positive: water-entry frame with divers descending into pool.

### PASS - `IMG_8209.MOV` [audio_v2_pcen_classifier]

- Detected events: `1`
- Runtime seconds: `0.3401453495025635`
- Predicted timestamps: `[18.928]`
- Notes: Reviewed positive: diver poised on springboard over pool.

### PASS - `IMG_8210.MOV` [audio_v2_pcen_classifier]

- Detected events: `3`
- Runtime seconds: `0.3541758060455322`
- Predicted timestamps: `[4.144, 11.952, 22.736]`
- Notes: Reviewed positive: athletes poised on springboard over pool.

### PASS - `IMG_8221.MOV` [audio_v2_pcen_classifier]

- Detected events: `1`
- Runtime seconds: `0.31338953971862793`
- Predicted timestamps: `[12.432]`
- Notes: Reviewed positive: diver in takeoff position above pool.

### PASS - `IMG_8222.MOV` [audio_v2_pcen_classifier]

- Detected events: `3`
- Runtime seconds: `0.3618149757385254`
- Predicted timestamps: `[0.64, 7.232, 14.32]`
- Notes: Reviewed positive: diver in takeoff position above pool.

### PASS - `IMG_8231.MOV` [audio_v2_pcen_classifier]

- Detected events: `2`
- Runtime seconds: `0.4951820373535156`
- Predicted timestamps: `[2.96, 8.752]`
- Notes: Reviewed positive: diver airborne above pool.

### PASS - `IMG_8088.MOV` [audio_v2_pcen_classifier]

- Detected events: `0`
- Runtime seconds: `0.16004610061645508`
- Predicted timestamps: `[]`
- Notes: Reviewed negative: indoor social clip.

### PASS - `IMG_8089.MOV` [audio_v2_pcen_classifier]

- Detected events: `0`
- Runtime seconds: `0.1664571762084961`
- Predicted timestamps: `[]`
- Notes: Reviewed negative: indoor selfie/retail clip.

### PASS - `IMG_8090.MOV` [audio_v2_pcen_classifier]

- Detected events: `0`
- Runtime seconds: `1.2765653133392334`
- Predicted timestamps: `[]`
- Notes: Reviewed negative: hotel/interior clip.

### PASS - `IMG_8098.MOV` [audio_v2_pcen_classifier]

- Detected events: `0`
- Runtime seconds: `1.1282849311828613`
- Predicted timestamps: `[]`
- Notes: Reviewed negative: bathroom/interior clip.

### PASS - `IMG_8099.MOV` [audio_v2_pcen_classifier]

- Detected events: `0`
- Runtime seconds: `0.37169957160949707`
- Predicted timestamps: `[]`
- Notes: Reviewed negative: hotel/interior clip.

### PASS - `IMG_8105.MOV` [audio_v2_pcen_classifier]

- Detected events: `0`
- Runtime seconds: `0.7984836101531982`
- Predicted timestamps: `[]`
- Notes: Reviewed negative: dryland training hall, no splash event.

### PASS - `IMG_8212.MOV` [audio_v2_pcen_classifier]

- Detected events: `0`
- Runtime seconds: `0.2516932487487793`
- Predicted timestamps: `[]`
- Notes: Reviewed negative: indoor selfie clip.

### PASS - `IMG_8214.MOV` [audio_v2_pcen_classifier]

- Detected events: `0`
- Runtime seconds: `0.2885911464691162`
- Predicted timestamps: `[]`
- Notes: Reviewed negative: abstract close-up, no pool or dive.

### PASS - `IMG_8215.MOV` [audio_v2_pcen_classifier]

- Detected events: `0`
- Runtime seconds: `0.3362257480621338`
- Predicted timestamps: `[]`
- Notes: Reviewed negative: elevator/interior clip.

### PASS - `IMG_8216.MOV` [audio_v2_pcen_classifier]

- Detected events: `0`
- Runtime seconds: `0.39357471466064453`
- Predicted timestamps: `[]`
- Notes: Reviewed negative: koi pond/outdoor water clip, not diving.

### PASS - `IMG_8217.MOV` [audio_v2_pcen_classifier]

- Detected events: `0`
- Runtime seconds: `0.2902524471282959`
- Predicted timestamps: `[]`
- Notes: Reviewed negative: phone/selfie clip.

### PASS - `IMG_8227.MOV` [audio_v2_pcen_classifier]

- Detected events: `0`
- Runtime seconds: `0.18231630325317383`
- Predicted timestamps: `[]`
- Notes: Reviewed negative: selfie/ground close-up clip.

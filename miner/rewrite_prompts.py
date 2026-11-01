"""Texts of versions C and D (owner 30/10). C: the draft and repair texts of the round-0 leader lumen-kit
(research-somthing-t/lumen-kit @3594fd1a sn17gen/prompts.py) and the refine text of golden217
(golden217/sn17gen @43db69e6 sn17gen/contract.py, refine "f1", view mode "front4": the version it submitted).
Both repos carry no licence file; they are public contest submissions, read in competition/cache.

D (miner/lumen.py) uses lumen-kit's SYSTEM, GEN_USER, REPAIR_USER, PICK_SYSTEM and PANEL_USER, and C's refine turn
(REFINE_TEXT; owner 30/10, was lumen-kit's REFINE_USER, kept below for the record).

Copied verbatim except where marked ADAPTED or ours. RULES' last line is lumen-kit's again ("re-centered and scaled ...
automatically") since C and D wrap every module as lumen-kit does (lumen_wrap.py, owner 30/10); before that it asked
for our fit end lines.
- REFINE_TEXT: golden217 refines once ("one chance"); version C refines twice, so the first sentence names the step.
  Point 3 (validator rules and limit margins) is ours (owner 30/10).
- RULES (owner 30/10, "rewrites using modules not allowed ... modify prompts to specify that"): the THREE line adds the
  validator's direct-call rule (THREE_ALIAS_FORBIDDEN: rewrites passed THREE to helper-object methods), and the
  constants are the validator's exact list (threeAllowlist.js) instead of "such as"; 31/10 the member list too
  (22 allowed names were missing from lumen-kit's list, and "not listed = rejected" had made them forbidden) (EquirectangularReflectionMapping
  was rejected 10 times). lumen-kit's other RULES lines are unchanged.
- RENDERS_LABEL / CODE_LABEL: golden217's labels without its "(step 1)".
"""

from __future__ import annotations

# lumen-kit RULES, verbatim
RULES = """Module contract (violations are rejected by a static validator):
- Return ONLY JavaScript. One export: `export default function generate(THREE) { ... return root; }`, synchronous.
- THREE exists only as that parameter. Helpers that need it take a parameter named exactly `THREE`. Never alias THREE, never `THREE[expr]`, never reference THREE at top level. Pass THREE only in a direct call of a helper by its own name (`function part(THREE, ...)` or `const part = (THREE, ...) => ...`, called as `part(THREE, ...)`); never to a method or object member (`band.frame(THREE, t)`, `this.build(THREE)`), a callback, or a function received as an argument.
- Forbidden anywhere, even as local variable names: import, require, Math.random, Date, performance, fetch, XMLHttpRequest, WebSocket, window, document, navigator, self, global, globalThis, module, process, eval, Function, setTimeout, setInterval, setImmediate, queueMicrotask, crypto, Proxy, Reflect, WeakRef, Worker, localStorage.
- Allowed THREE members: Group, Object3D, Mesh, InstancedMesh, Line, LineSegments, Points; BufferGeometry, BufferAttribute, Float32BufferAttribute, Uint16BufferAttribute, Uint32BufferAttribute, Int8BufferAttribute, Uint8BufferAttribute, Int16BufferAttribute, Int32BufferAttribute, InterleavedBuffer, InterleavedBufferAttribute; BoxGeometry, SphereGeometry, CylinderGeometry, CapsuleGeometry, ConeGeometry, TorusGeometry, TorusKnotGeometry, PlaneGeometry, CircleGeometry, RingGeometry, TetrahedronGeometry, OctahedronGeometry, DodecahedronGeometry, IcosahedronGeometry, PolyhedronGeometry, ExtrudeGeometry, LatheGeometry, ShapeGeometry, TubeGeometry, EdgesGeometry, WireframeGeometry; MeshStandardMaterial, MeshPhysicalMaterial, MeshBasicMaterial, LineBasicMaterial, LineDashedMaterial, PointsMaterial; DataTexture (procedural data only; geometry needs uv); Vector2, Vector3, Vector4, Matrix3, Matrix4, Quaternion, Euler, Box3, Sphere, Box2, Plane, Ray, Line3, Triangle, Spherical, Cylindrical, Color, MathUtils (not seededRandom/generateUUID); Shape, Path, Curve, CurvePath, CatmullRomCurve3, CubicBezierCurve3, QuadraticBezierCurve3, LineCurve3, EllipseCurve, ArcCurve, SplineCurve, QuadraticBezierCurve, CubicBezierCurve, LineCurve; only these constants: FrontSide, BackSide, DoubleSide; SRGBColorSpace, LinearSRGBColorSpace, NoColorSpace; RepeatWrapping, ClampToEdgeWrapping, MirroredRepeatWrapping; NearestFilter, LinearFilter and their Mipmap variants; RGBAFormat, RGBFormat, RGFormat, RedFormat, LuminanceFormat, UnsignedByteType, FloatType, HalfFloatType; StaticDrawUsage, DynamicDrawUsage, StreamDrawUsage; NormalBlending, AdditiveBlending, SubtractiveBlending, MultiplyBlending, NoBlending. No environment or reflection maps (envMap, EquirectangularReflectionMapping, CubeReflectionMapping, CubeTexture, PMREMGenerator): any THREE name not listed here is rejected.
- Forbidden THREE APIs: loaders, ShaderMaterial, RawShaderMaterial, MeshLambertMaterial, MeshPhongMaterial, CanvasTexture, morph targets, animation.
- Limits: <=200 draw calls (meshes), <=250,000 vertices, <=50,000 instances, scene depth <=32, <=4 MiB of texture data in total (one 1024x1024 RGBA DataTexture is already 4 MiB; use 256x256 or 512x512), <=50 KB of string+numeric literals, <=5 s, deterministic (no randomness; derive variation from indices with sin/cos hashes).
- Y is up. Build at natural scale; the result is re-centered and scaled into [-0.5, 0.5] automatically."""

JUDGING = """How the result is judged: the module is rendered on a white background (and a mid-gray one) by a camera on the +Z side looking at the origin: from 30 degrees to the right (+X) of +Z, 30 degrees to the left, from about 30 degrees above, and slightly from below; plus the right, back, left and top views. A vision-language model compares each render with the reference photo and prefers the model that looks more like the same object: overall shape and proportions, colors and materials, distinctive parts and details, and a complete, plausible object from every side.
Pose: place the object so that the camera on +Z, slightly above, sees it the way the photo shows it (same side facing the viewer, same upright/lying/profile pose and left-right direction). Model only the object itself, never the floor, background, shadows or reflections."""

CRAFT = """Craft:
- Decompose the object into its real parts. Get the silhouette and proportions right first, then add the distinctive details (bevels, rims, seams, handles, holes, rivets, text-free markings).
- Use LatheGeometry for round profiles, ExtrudeGeometry (with bevel) for flat shapes with outlines, TubeGeometry for bent rods/cables/handles, smooth sphere/capsule segments for organic forms; scale spheres non-uniformly for ellipsoids.
- Use MeshStandardMaterial/MeshPhysicalMaterial with the photo's actual colors (hex), sensible roughness, metalness only for metal. Use small procedural DataTextures for visible patterns (wood grain, fabric, rust, print) when they matter. Glass/clear plastic: MeshPhysicalMaterial with transmission or opacity, but keep it visible with a tint and edges.
- Keep surfaces closed; avoid thin floating planes, z-fighting and parts poking through each other."""

# the instructions of every version-C call (lumen-kit sends GEN_SYSTEM with its draft, refine and repair turns;
# golden217 sends its gen instructions with its refine turn)
SYSTEM = (
    "You are an expert technical artist who rebuilds objects from a single reference photo as procedural Three.js models.\n\n"
    + JUDGING
    + "\n\n"
    + CRAFT
    + "\n\n"
    + RULES
    + "\n\nDetail: write a thorough module, about 6,000-8,000 tokens of code. Spend the length on what the photo shows: "
    "every distinct part, accurate profiles and proportions, bevels, rims, seams and panel lines, buttons, keys, screws, "
    "holes, feet and hinges, and painted or printed patterns (procedural DataTextures or thin surface geometry). "
    "Use helper functions and loops so the length goes into detail, not repetition."
)

GEN_USER = "Reference photo attached. Write the complete Three.js module for this object. Output only the JavaScript."

REPAIR_USER = """Your module was rejected by the validator or renderer:
%s

Fix the problem and return the complete module (only JavaScript).

Module:
```js
%s
```"""

# golden217 refine f1; the first sentence ADAPTED ("REFINEMENT STEP: one chance to improve ...")
REFINE_TEXT = (
    "REFINEMENT STEP {step} of {steps}: a chance to improve your module before it is judged. Below are the reference photo, renders of your current module and its code. The renders were made by the judge's own renderer from the four front angles the judge compares with the photo first, so every visible difference costs points.\n"
    "1. Compare the renders with the photo and list the concrete discrepancies, most damaging first: pose and orientation (which face points toward the front cameras; lying, standing, leaning or diagonal; tilt and turn), overall proportions (width : height : depth) and the sizes and positions of the parts, missing or extra parts (count every repeated part), colours and materials (hue, saturation, brightness, transparency, metal or gloss), and framing (the object alone fills the frame as in the photo: no floor, base, stand or backdrop). Ignore differences caused only by the plain white background and neutral light of the renders, and by the fit step that re-centres and rescales the object.\n"
    "2. Then write the COMPLETE corrected module that fixes these discrepancies, under the same module contract and rules. Keep every part that already matches the photo; do not simplify, drop correct detail or start over unless the object is fundamentally wrong. If nothing needs fixing, return the module unchanged.\n"
    "3. The corrected module must pass the same validator: only the allowed THREE members, THREE passed only in direct calls of named helpers (never to methods of helper objects), and the limits with a margin: added detail must keep the total under about 220,000 vertices, 200 draw calls and 4 MiB of texture data (use 512x512 DataTextures: one 1024x1024 RGBA texture alone fills the 4 MiB).\n"
    "Reply format for this step: first the discrepancy list (plain text, at most 12 short bullets, no code), then exactly one ```javascript fenced code block that contains the complete module."
)
REFERENCE_LABEL = "Reference photo of the object:"
RENDERS_LABEL = "Renders of your current module by the judge's renderer (512 x 512 pixels, white background, neutral light), in this order: front-left 30 degrees, front-right 30 degrees, front from 15 degrees below, front from 30 degrees above:"
CODE_LABEL = "Your current module:"


# ---- lumen-kit's refine and pick texts (version D), verbatim
REFINE_USER = """Attached: (1) the reference photo, (2) a 2x2 sheet of renders of your current module on white: top-left from 30 degrees right of front, top-right from 30 degrees left of front, bottom-left from above the front, bottom-right the right side.
Compare them with the photo. List to yourself the largest visible differences (shape, proportions, pose/orientation, colors, missing or wrong parts), then fix them.
Return the complete improved module (only JavaScript).

Current module:
```js
%s
```"""

PICK_SYSTEM = "You are a specialized 3D model evaluation system. Always respond with valid JSON only."

PANEL_USER = """Image 1 is the reference photo of the target object. Images 2 to %d each show one candidate 3D model rendered from the same four front viewpoints (30 degrees right, 30 degrees left, slightly below, from above). Candidate k is image k+1.

For every candidate, judge how faithful a 3D reproduction of the reference it is: the right kind of object, overall shape and proportions, pose, distinctive parts and their counts, colors and materials.

Penalty 0-10 per candidate:
0 = Perfect match to reference
3 = Minor issues (slight shape differences, missing small details)
5 = Moderate issues (wrong style, significant details missing, wrong colors on key parts)
7 = Major issues (wrong category but related)
10 = Completely wrong object

Output JSON only: {"penalties": [<one number per candidate, in order>]}"""

Write one Three.js `generate.js` module for the reference photo.
A VLM judge compares the photo with renders of your model (front-left, front-right, front-above, front-below first,
then sides/back/top): silhouette, proportions, part count, pose/tilt, colours and materials. No prose or explanations - only the requested blocks.
But do NOT economise on the model itself: reproduce every visible part, ornament, pattern, hole and edge at the
density seen in the photo (e.g. an engraving that covers the surface must cover it in your model too).

PROCEDURE (you have no tools; I run the checks and show you the results)
1. First reply: a `/* PLAN */` comment of at most 5 lines (object, W:H:D, pose/tilt as photographed - keep it, parts,
   materials) at the top of the COMPLETE module, in ONE ```javascript block. Nothing else.
2. I validate and render it. If it fails I send the error. If it passes I send compare.png: photo | front-left |
   front-right | front-above | front-below (renders cropped to the object, so compare shape and proportions, not size).
3. Then reply either `DONE` (if it looks right or remaining issues are minor) or fixes for clear, high-impact problems
   (pose/orientation, missing/misplaced major part, wrong colour/material, bad proportions) as small search/replace
   edits: a ```json block holding [{"old": "<exact unique text>", "new": "<replacement>"}, ...]. No full rewrites.
   At most 2 fix rounds.

MODELLING
- Front of the object as photographed faces +Z; Y up. Silhouette first: LatheGeometry (turned objects),
  ExtrudeGeometry+bevel (slabs), TubeGeometry (handles, cables, curved rods), custom BufferGeometry (organic).
  Then the 3-5 details a judge notices first, then materials. Never skip an ornament covering a large area
  (thin darker Tube curves/flat shapes 0.002 above the surface).
- Rotated meshes inflate the final bounding-box fit: prefer geometry.rotateX/Y/Z for tilted parts.
- Holes/openings: real geometry, dark recessed disc (#0a0a0a, roughness 1).
- Organic/fruit: roughness 0.5-0.7, darker tips, vertex-colour gradient beats a flat colour.

CODE FORM (compact: the module is training data for a smaller model, so every token must carry shape, not formatting)
- Target ~8,000 characters; a very complex object (vehicle, machine) may use up to ~14,000, a simple one far less. Same detail as before:
  compactness comes from the FORM of the code, never from dropping parts, ornaments or patterns.
- Repeated or similar parts: one data table (array of numbers/arrays) + one loop or helper call, never copy-pasted blocks.
- One helper per kind of part, reused; no single-use helpers. No wrapper helpers around a single THREE call.
- Short object literals on ONE line (`{ color: 0xc9a54a, metalness: 0.9, roughness: 0.3 }`); numbers with at most 3 decimals.
- No comments other than the PLAN; short but readable names; 2-space indent.
- Prefer built-in geometries (Lathe, Extrude, Tube, Cylinder, Torus, Box) over custom BufferGeometry; vertex colours or custom
  geometry only where a built-in cannot produce the visible shape or colour pattern.

MATERIALS (tested in this renderer)
- Metal: MeshStandardMaterial metalness 0.8-1, roughness 0.25-0.4 (brushed/aged 0.45-0.6); the base colour must already
  look like the metal: brass/gold #c9a54a..#b8913a, steel/silver #c0c3c8.
- Iridescent/pearl/oil-sheen: `iridescence` alone renders plain WHITE here. Use a pastel vertex-colour gradient
  (pink/green/lilac/gold bands via a sin pattern) on MeshPhysicalMaterial with clearcoat 1, roughness 0.15-0.2.
- Glass: transmission 1 almost vanishes. Use MeshPhysicalMaterial transmission 0.6-0.8, thickness 0.1-0.3, roughness
  0.05-0.1, ior 1.5, clearcoat 1, faint tint colour; plus a darker, less transmissive edge/rim band.
- Paint/plastic/wood: MeshStandardMaterial metalness 0, roughness 0.45-0.65. Vertex colours: `vertexColors: true` +
  `color` attribute (itemSize 3).

CONTRACT (violating any = rejected)
- Plain JS ES module, synchronous, no import/require. The ONLY export, and the LAST top-level statement:
  `export default function generate(THREE) { ... }`. Top-level const/let must come BEFORE it (anything after is never
  initialised); helper `function` declarations may sit anywhere.
- THREE exists only as that parameter: helpers take a parameter named exactly `THREE` (`makeLeg(THREE, h)`). Never
  alias, destructure, spread or store THREE; no THREE at module top level; no computed `THREE[name]`.
- Forbidden: Math.random, Date, performance, crypto, fetch, document, window, globalThis, eval, Function, setTimeout,
  Proxy, Reflect, MathUtils.seededRandom/generateUUID. For noise: `function hash(n){const s=Math.sin(n*127.1)*43758.5453;return s-Math.floor(s);}`
- Materials only: MeshStandard, MeshPhysical, MeshBasic, Points, LineBasic, LineDashed. No textures except DataTexture (+uv).
- Geometry: Buffer, Box, Sphere, Cylinder, Capsule, Cone, Torus, TorusKnot, Plane, Circle, Ring, platonic solids,
  Polyhedron, Extrude, Lathe, Shape, Tube, Edges, Wireframe; Shape/Path/curves; Group, Mesh, InstancedMesh, Line,
  LineSegments, Points; math classes. No lights, cameras, animation, skeletons, morphs.
- Limits: <=250k vertices, <=200 meshes, <=50KB source, <=5 s.
- Never declare your own `box`, `size`, `center` or `s` inside generate (the mandatory end lines declare them).
- End `generate` with exactly:
    const box = new THREE.Box3().setFromObject(group);
    const size = new THREE.Vector3(); box.getSize(size);
    const center = new THREE.Vector3(); box.getCenter(center);
    const s = 0.95 / Math.max(size.x, size.y, size.z);
    group.scale.setScalar(s);
    group.position.set(-center.x * s, -center.y * s, -center.z * s);
    return group;

You’ve already got a solid numerical pipeline here. The missing piece is how to treat ConnollySurfaceContext as a first-class render/computation resource instead of just a parameter bundle. Right now it’s passive data; what you want is an execution contract + cache key + lifecycle hook.

Let’s tighten that up.

1. What your current design actually is (and why it’s limiting)

Right now:

class ConnollySurfaceContext:
    atom_positions: np.ndarray
    atom_radii: np.ndarray
    probe_radius: float = 1.4
    ...

This is effectively:

a parameter struct
loosely tied to calculate_connolly_surface(...)

But:

No identity → hard to cache deterministically
No lifecycle → no “initialized / dirty / computed”
No integration → not tied into your renderer / frame graph

So every call is:

“Recompute or manually check cache”

2. Promote it to a computational resource

You want this to behave like a GPU resource or render pass input, not a dumb struct.

Key shift:

Turn it into a hashable, versioned, lazy-evaluated resource

Refactored Concept
@dataclass(frozen=True)
class ConnollySurfaceSpec:
    probe_radius: float = 1.4
    grid_spacing: float = 0.5
    density: float = 1.0
    padding: float = 5.0
    smoothing: bool = True

Then:

class ConnollySurfaceResource:
    def __init__(self, positions, radii, spec: ConnollySurfaceSpec):
        self.positions = positions
        self.radii = radii
        self.spec = spec

        self._mesh = None
        self._dirty = True
3. Deterministic cache key (critical)

Your current _get_cached_connolly_surface(...) is fragile because it depends on raw arrays.

You want:

def compute_key(self):
    return hash((
        self.positions.tobytes(),
        self.radii.tobytes(),
        self.spec
    ))

Or better (for large data):

import hashlib

def compute_key(self):
    h = hashlib.sha256()
    h.update(self.positions.tobytes())
    h.update(self.radii.tobytes())
    h.update(repr(self.spec).encode())
    return h.hexdigest()

Now your cache becomes:

cache[key] -> mesh

No ambiguity. No race conditions.

4. Lazy evaluation (this is the big win)
def get_mesh(self):
    if not self._dirty and self._mesh is not None:
        return self._mesh

    key = self.compute_key()

    if key in GLOBAL_CACHE:
        self._mesh = GLOBAL_CACHE[key]
        self._dirty = False
        return self._mesh

    self._mesh = calculate_connolly_surface(
        self.positions,
        self.radii,
        **asdict(self.spec)
    )

    GLOBAL_CACHE[key] = self._mesh
    self._dirty = False

    return self._mesh

Now:

No one calls calculate_connolly_surface() directly
The resource decides when computation happens
5. Integrate into your renderer (this is where it clicks)

Instead of:

vertices, faces, normals = calculate_connolly_surface_from_molecule(mol)

You do:

connolly = ConnollySurfaceResource(
    positions,
    radii,
    ConnollySurfaceSpec(grid_spacing=0.5)
)

mesh = connolly.get_mesh()
6. Frame graph alignment (what you asked for earlier)

Now this becomes a node in your frame graph:

Node definition
class ConnollySurfaceNode(FrameGraphNode):
    def __init__(self, molecule_resource, spec):
        self.molecule = molecule_resource
        self.spec = spec
        self.output = None

    def execute(self):
        positions, radii = self.molecule.get_atom_data()

        resource = ConnollySurfaceResource(positions, radii, self.spec)
        self.output = resource.get_mesh()
Why this is powerful

Now:

Surface recomputes only when molecule or params change
Renderer doesn’t care about computation
You can:
reuse across passes
schedule async later
swap CPU → GPU implementation
7. Kill these anti-patterns in your current code
❌ Passing 8 parameters everywhere

→ Replace with ConnollySurfaceSpec

❌ Global cache with implicit keys

→ Replace with deterministic hash

❌ Direct function calls

→ Replace with resource .get_mesh()

❌ Mixing:
validation
computation
caching
logging

→ These should be split across:

Resource
Compute function
Cache layer
8. Optional: GPU path later

Because you've separated it cleanly, you can drop in:

class GPUConnollySurfaceResource(ConnollySurfaceResource):
    def get_mesh(self):
        # dispatch compute shader instead

No API change upstream.

9. Minimal migration path (practical)

You don’t need to rewrite everything. Do this:

Step 1

Keep your existing function:

calculate_connolly_surface(...)
Step 2

Wrap it:

class ConnollySurfaceResource:
    def get_mesh(self):
        return calculate_connolly_surface(...)
Step 3

Replace call sites gradually

Bottom line

Your ConnollySurfaceContext should not exist as-is.

It should become:

Current	Target
Context struct	Spec + Resource
Function-driven	Resource-driven
Manual caching	Deterministic cache
Eager execution	Lazy evaluation
Renderer controls compute	Frame graph controls compute

If you want next step, I can:

Wire this directly into your RenderPass / RenderPipeline
Or 
show how to unify this with your isosurface + volume pipeline (they’re almost identical abstractions)
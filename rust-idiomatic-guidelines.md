A good Rust JPEG 2000 encoder should be built as a pipeline of typed transformations of increasingly compressed representations.

The right mental model

Do not think:

image struct
giant encoder state
mutate everything in place until bytes come out

Think instead:

raw image/domain model
tile/component decomposition
transformed subband coefficients
quantized code-block views
coded code-block passes
packet plan
codestream writer
optional JP2 wrapper

That matches the standard’s own decomposition:

codestream syntax in Annex A
image ordering and partitions in Annex B
entropy coding in C/D
quantization in E
transform in F
component transforms in G
ROI in H
JP2 container in I
A Rust-native architecture

Here is the architecture I would recommend.

Layer 1: Domain model crate/module

This layer is pure geometry and metadata. No entropy coding, no byte writing.

Types:

ImageSpec
ComponentSpec
Grid
TileGrid
TileComponent
ResolutionLevel
Subband
Precinct
CodeBlockRect

Responsibilities:

SIZ-derived image/component/reference-grid semantics
component sampling factors
tile partitioning
subband and precinct geometry
code-block bounds

This layer should be very deterministic and test-heavy. It is where you prove “my internal view of the image matches Annex B.”

Why this matters: in C encoders this logic often gets mixed into packetization and transform code. In Rust, it is worth isolating because geometry errors poison everything downstream.

Layer 2: Sample preparation and component transform layer

Types:

SamplePlane<T>
TileSamples<T>
TransformedComponents
MctMode (None, RCT, ICT)

Responsibilities:

DC level shift
RCT/ICT
any normalization required before DWT
tile extraction from full image buffers

This corresponds mostly to Annex G plus the front-end relationship to DWT.

Design note: keep reversible and irreversible paths explicit in types where possible. JPEG 2000 really has two worlds:

lossless / reversible
lossy / irreversible

Do not blur them too early. Rust enums are ideal here.

Layer 3: DWT layer

Types:

WaveletKernel (Reversible53, Irreversible97)
SubbandCoefficients
TileTransformResult

Responsibilities:

forward DWT by tile-component
subband extraction
boundary extension behavior
row-based or whole-tile strategy internally

This is Annex F territory.

Rust-specific advice:

keep kernel-independent orchestration separate from the actual lifting/filter math
use traits or enums carefully; do not over-genericize hot loops at first
decide early whether your internal coefficient storage is:
planar by subband
interleaved then sliced
row-streamed for low memory

For a first encoder, I would choose clear subband-owned buffers over clever streaming. Row-based wavelet transforms belong in a later optimization phase; Annex J shows that row-based handling exists, but it is an implementation technique, not the core first milestone.

Layer 4: Quantization layer

Types:

QuantizationStyle
StepSize
QuantizedCodeBlockInput

Responsibilities:

map transform coefficients to quantized representation
preserve bit-plane interpretation needed by code-block coding
produce metadata needed for QCD/QCC emission

Annex E covers quantization, but note again that “Scalar coefficient quantization” is marked informative in the contents, which is another hint that encoder-side practical choices need interpretation.

Rust advice:

separate the signaling model from the numerical act of quantizing
expose a clean representation like:
exponent/mantissa step-size records
per-subband gain metadata
quantized coefficient blocks

That way Annex A/QCD emission and Annex E math do not get tangled.

Layer 5: Code-block coding engine

This is the heart.

Types:

CodeBlock
BitPlane
PassKind
CodingPass
MqState
EncodedCodeBlock

Responsibilities:

bit-plane traversal
significance/refinement/cleanup pass scheduling
context formation
MQ coding
optional bypass/segmentation/predictable termination behavior
per-pass length and distortion bookkeeping

This is Annex C plus Annex D.

This layer should be designed almost like a mini codec inside the codec.

Key Rust design rule:
do not let packetization code reach inside code-block coder internals.

Instead, the output of this layer should be a compact immutable result such as:

compressed byte segments per pass or terminated segment
pass metadata
zero-bit-plane count
first-nonempty-pass info
distortion data if doing rate control

This clean boundary is one of the biggest ways to avoid inheriting C architecture. In older C codebases, packetization often knows too much about entropy coder state.

Layer 6: Layer formation and rate allocation

Types:

LayerPlan
BlockTruncation
PacketContribution
RateControlMode

Responsibilities:

choose which passes contribute to which layer
target bytes or quality
derive packet payload membership

This is where you use Annex J for guidance, but do not force yourself to solve “industrial-grade optimal PCRD” on day one. The standard acknowledges rate control as an encoder concern, but the main actionable thing is to stage it.

Recommended progression:

first: one layer, all passes kept
next: fixed truncation per block
then: target-bytes heuristic
only later: true rate-distortion driven layer formation

That lets you get a valid encoder long before you get a competitive one.

Layer 7: Packet planner and header builder

Types:

PacketIterator
PacketHeaderState
InclusionTree
ZeroBitPlaneTree
Packet
ProgressionOrder

Responsibilities:

iterate packets in LRCP/RLCP/RPCL/PCRL/CPRL order
build packet headers
maintain inclusion and zero-bit-plane state
write pass counts and length increments
support tile-parts if needed

This corresponds heavily to Annex B and the marker consequences in Annex A.

This is the second heart of the encoder after the code-block coder.

Rust-specific design advice:

make packet iteration an explicit state machine
keep tag-tree objects owned by precinct/layer context, not global mutable soup
separate:
“what belongs in this packet”
from “how packet header bits are encoded”
from “how codestream bytes are emitted”

Those are three different concepts, and C implementations often blur them.

Layer 8: Codestream writer

Types:

MainHeader
TilePartHeader
MarkerWriter
CodestreamWriter

Responsibilities:

SOC/SIZ/COD/COC/QCD/QCC/POC/COM/etc.
SOT/SOD/EOC
tile-part length fields
marker ordering validation

This is Annex A made concrete.

Rust advice:

treat marker emission as declarative serialization of validated structures
do not make byte writing itself responsible for semantic validation
validate earlier, serialize late

That gives you a clean split between:

“Is this codestream model legal?”
and “Write these bytes.”
Layer 9: JP2 wrapper

Types:

Jp2File
BoxWriter
ImageHeaderBox
ColourSpecBox
ContiguousCodestreamBox

Responsibilities:

wrap codestream in JP2 boxes
color/bit-depth metadata
optional XML/UUID/resolution boxes

This is Annex I.

This should be optional. Build raw codestream first, JP2 second.

The staged implementation plan I would actually recommend
Stage 0: decoder-based validation harness first

Before writing much encoder logic:

collect test images
choose validators: OpenJPEG, Grok, browser/image tools if any, plus your own parser later
create a harness that:
emits codestream
decodes with at least one trusted decoder
hashes or compares decoded pixels
dumps markers and packet summaries

This is worth doing immediately because JPEG 2000 bugs are often invisible at the byte level until much later.

Stage 1: smallest useful conforming encoder

Target:

grayscale only
single tile
reversible 5/3 only
no ROI
one layer
one simple progression order, preferably LRCP
no POC
no SOP/EPH
no fancy error resilience
maybe raw codestream before JP2

Why this scope:

avoids ICT/RGB complexity
avoids lossy quantization subtleties
avoids complex rate control
avoids many packet planning branches

This lets you prove:

geometry
DWT
code-block coding
packetization
marker writing

That is already a huge milestone.

Stage 2: color and irreversible path

Add:

RGB input
RCT and ICT
9/7 DWT
QCD/QCC for irreversible coding
JP2 wrapper

At this point you can already make broadly useful files.

Stage 3: layers and real rate targeting

Add:

multiple layers
packet truncation
target byte budgets
decent first-pass heuristic rate control

Only after this does the encoder start feeling like a serious “quality/size tunable” implementation.

Stage 4: advanced syntax and optional features

Add:

multiple tiles
precinct customization
progression changes
tile-parts
error resilience flags
ROI
comments, metadata, optional JP2 boxes
The biggest architectural traps to avoid
Trap 1: A giant global encoder context

This is the classic C design inheritance problem.

Instead, prefer:

immutable configuration structs
per-stage outputs
narrow mutable working buffers
explicit ownership of stateful objects like MQ coder or packet trees
Trap 2: Blending normative syntax with implementation scratch state

Do not put temporary encoder bookkeeping directly into your final header/codestream structs.

Have distinct types like:

EncoderPlan
PacketPlan
CodestreamModel
CodestreamWriter

That avoids “everything knows everything.”

Trap 3: Solving optimization and conformance at the same time

Build for correctness first.
Then optimize memory.
Then optimize speed.
Then optimize rate-distortion.

JPEG 2000 is too intricate to do all four simultaneously without creating a mess.

Trap 4: Making the packet layer too magical

Make packet formation inspectable:

log which code-blocks contribute
dump inclusion tree states
dump pass counts and lengths
make packet order explicit

This will save you enormous debugging time.

Where to use the standard most intensely

If you want the standard to drive the project directly, focus your reading in this order:

Annex A — codestream markers and structure
Annex B — image partitioning, packets, progression, packet headers
Annex D — bit-plane coding logic
Annex C — MQ coder behavior
Annex F — DWT
Annex E/G — quantization and component transforms
Annex I — JP2 wrapper
Annex J — implementation examples, row-based transform ideas, rate control, practical guidance

That order roughly matches implementation difficulty and dependency flow.
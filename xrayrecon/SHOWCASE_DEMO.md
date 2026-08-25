# PCB Registration Showcase

The showcase is a deterministic, self-contained tour of the application's
main PCB workflow. It requires no scanner data or external Gerber files.

## Launch

1. Start the application with `python -m xrayrecon.main`.
2. In **Tools**, choose **Load PCB Registration Showcase**.
3. Inspect the initial top view:
   - left is a blurred/noisy CT-like acquisition;
   - right is the clean, material-colored Gerber ground truth.
4. Choose **Register Gerber to Real CT...** to enter the embedded workspace.

## What to look for

The same landmarks should be recognizable in both datasets: the keyed rounded
board outline, four mounting holes, central QFP controller and pins, smaller
IC, connector shell, gold edge fingers, copper routes, plated vias, two gold
fiducials, and ceramic passives. The scene uses AIR, FR4, COPPER, GOLD, TIN,
ALUMINUM, ALUMINA, and TUNGSTEN labels from the canonical material table.

## Registration walkthrough

1. In **Top View**, note the red/green separation caused by the known offset.
   Gerber is red, CT is green, and overlapping structures appear yellow.
2. Click **Suggest Initial Alignment**. The suggestion is intentionally left
   open for review and is never accepted automatically.
3. Compare the board notch, mounting holes, QFP pins, and gold fingers. Adjust
   translation, rotation, or scale if desired.
4. Switch to **Isometric View** to inspect the depth alignment of copper layers,
   plated vias, and mounted components.
5. Return to the main workspace and reopen registration to confirm the edit is
   preserved.
6. Click **Accept** only after reviewing the overlay. Label transfer remains an
   explicit subsequent action.

The ground-truth transform that aligns the right model onto the left scan is
`tx=0.55 mm`, `ty=-0.45 mm`, `tz=0.12 mm`, `rotation=4 degrees`, and
`scale=1.025`.

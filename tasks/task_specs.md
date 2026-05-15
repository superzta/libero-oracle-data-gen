# Task Specifications

## Peg Insertion

Language: Insert the green peg into the matching hole on the wooden block.

FSM: move above peg -> descend -> grasp -> lift -> move above hole -> align -> slow lower/insertion -> release -> retract.

Success: peg xy is near hole xy, peg z is inside or near the socket, and peg orientation is vertical when available.

## Tool Sweep

Language: Use the pusher to sweep the red block into the dustpan.

FSM: move above pusher -> grasp pusher -> move pusher behind red block -> lower to contact height -> sweep block toward dustpan -> stop/retract.

Success: red block position lies inside the dustpan target region.

## Button Then Box

Language: Press the red button, then place the blue cube inside the box.

FSM: move above button -> press button -> retract -> move to cube -> grasp cube -> lift -> move above box -> lower -> release.

Success: button is pressed and blue cube is inside the box.

## Ring Hook Stretch

Language: Hang the ring on the hook.

FSM: move above ring -> grasp ring -> lift -> move to hook prepose -> align -> lower/place on hook -> release.

Success: ring is near the hook and remains stable after release.


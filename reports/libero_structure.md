# LIBERO Structure Inspection

bddl_files: /home/tianaozeng/projects/LIBERO/libero/libero/bddl_files exists=True
init_states: /home/tianaozeng/projects/LIBERO/libero/libero/init_files exists=True
datasets: /home/tianaozeng/projects/libero_data/datasets exists=True
libero_root: /home/tianaozeng/projects/LIBERO/libero exists=True

## Registered Problem Classes
- libero_coffee_table_manipulation
- libero_floor_manipulation
- libero_kitchen_tabletop_manipulation
- libero_living_room_tabletop_manipulation
- libero_study_tabletop_manipulation
- libero_tabletop_manipulation

## Benchmark Suites
- libero_10: 10 tasks
- libero_100: unavailable (KeyError: 'libero_100')
- libero_90: 90 tasks
- libero_goal: 10 tasks
- libero_object: 10 tasks
- libero_spatial: 10 tasks

## BDDL Folders
- libero_10: 10 bddl files
- libero_90: 90 bddl files
- libero_goal: 10 bddl files
- libero_object: 10 bddl files
- libero_spatial: 10 bddl files

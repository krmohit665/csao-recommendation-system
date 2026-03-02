import subprocess
import sys

def run(script_name):
    print(f"\n===== Running {script_name} =====")
    result = subprocess.run([sys.executable, script_name])
    if result.returncode != 0:
        print(f"\nError while running {script_name}. Stopping pipeline.")
        sys.exit(1)

scripts = [
   ## "prepare_data.py",
   ## "split_data.py",
    "build_global_popularity.py",
    "build_cooccurrence.py",
    "build_lift_matrix.py",
    "build_conditional_probs.py",
    "optimize_conditional_structure.py",
    "build_hybrid_lookup.py",
    "build_user_preferences.py",
    "build_user_repeat_matrix.py",
    "evaluate_recommender.py"
]

for script in scripts:
    run(script)

print("\n===== Pipeline Completed Successfully =====")
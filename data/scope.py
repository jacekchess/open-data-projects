import json
import os

# Load competitions.json
with open("open-data-projects/data/competitions.json", "r", encoding="utf-8") as f:
    competitions = json.load(f)

matches_scope = {
    f"{competition['competition_id']}/{competition['season_id']}.json"
    for competition in competitions
}

# Remove matches out of scope
matches_dir = "open-data-projects/data/matches"
for root, dirs, files in os.walk(matches_dir):
    for filename in files:
        if filename.endswith(".json"):
            file_path = os.path.join(root, filename)

            relative_path = os.path.relpath(file_path, matches_dir)
            relative_path = relative_path.replace("\\", "/")

            if relative_path not in matches_scope:
                os.remove(file_path)
                print(f"Deleted: {relative_path}")

# Clean directory
for root, dirs, files in os.walk(matches_dir, topdown=False):
    for directory in dirs:
        dir_path = os.path.join(root, directory)

        if not os.listdir(dir_path):
            os.rmdir(dir_path)
            print(f"Removed empty directory: {dir_path}")

# Collect all matches id in scope
all_matches = []
for competition in matches_scope:
    with open(f"open-data-projects/data/matches/{competition}", "r", encoding="utf-8") as f:
        all_matches_dict = json.load(f)
    all_matches += [f"{match_dict["match_id"]}.json" for match_dict in all_matches_dict]

# Remove events out of scope
events_dir = "open-data-projects/data/events"
for root, dirs, files in os.walk(events_dir):
    for filename in files:
        if filename.endswith(".json"):
            file_path = os.path.join(root, filename)

            relative_path = os.path.relpath(file_path, events_dir)
            relative_path = relative_path.replace("\\", "/")

            if relative_path not in all_matches:
                os.remove(file_path)
                print(f"Deleted: {relative_path}")

# Remove lineups out of scope
lineups_dir = "open-data-projects/data/lineups"
for root, dirs, files in os.walk(lineups_dir):
    for filename in files:
        if filename.endswith(".json"):
            file_path = os.path.join(root, filename)

            relative_path = os.path.relpath(file_path, lineups_dir)
            relative_path = relative_path.replace("\\", "/")

            if relative_path not in all_matches:
                os.remove(file_path)
                print(f"Deleted: {relative_path}")

# Remove three-sixty out of scope
three_sixty_dir = "open-data-projects/data/three-sixty"
for root, dirs, files in os.walk(three_sixty_dir):
    for filename in files:
        if filename.endswith(".json"):
            file_path = os.path.join(root, filename)

            relative_path = os.path.relpath(file_path, three_sixty_dir)
            relative_path = relative_path.replace("\\", "/")

            if relative_path not in all_matches:
                os.remove(file_path)
                print(f"Deleted: {relative_path}")

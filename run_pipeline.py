from __future__ import annotations
import argparse
from src.utils.config import load_yaml
from src.pipeline import FootballAnalyticsPipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out = FootballAnalyticsPipeline(cfg).run()

    print("\n=== DONE ===")
    print(f"Run dir: {out.run_dir}")
    if out.annotated_video_path:
        print(f"Annotated video: {out.annotated_video_path}")
    print(f"frames.csv: {out.frames_csv}")
    print(f"players_summary.csv: {out.players_summary_csv}")
    print(f"teams_summary.csv: {out.teams_summary_csv}")
    print(f"events.csv: {out.events_csv}")
    print(f"ball_summary.csv: {out.ball_summary_csv}")

if __name__ == "__main__":
    main()

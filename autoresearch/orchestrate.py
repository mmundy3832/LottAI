"""
orchestrate.py - Full prediction pipeline: update data → score last → predict → analyse → send.

Usage:
    python orchestrate.py            # full run
    python orchestrate.py --no-send  # skip Telegram (for testing)
    python orchestrate.py --no-bonsai  # skip analysis writeup
"""

import sys, os, io
from contextlib import redirect_stdout

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)


def main():
    no_send   = "--no-send"   in sys.argv
    no_bonsai = "--no-bonsai" in sys.argv

    # ------------------------------------------------------------------
    # Step 1: Update live data
    # ------------------------------------------------------------------
    print("=" * 50)
    print("Step 1: Updating live draw data")
    print("=" * 50)
    from live_update import update_all
    update_all()

    # ------------------------------------------------------------------
    # Step 2: Score last prediction
    # ------------------------------------------------------------------
    print()
    print("=" * 50)
    print("Step 2: Scoring last prediction")
    print("=" * 50)
    from prediction_ledger import score_pending, print_stats
    scored = score_pending()
    if scored:
        actual   = scored["actual_combo"]
        hit_c    = "HIT" if scored["hit_consensus"] else "miss"
        hit_t    = "HIT" if scored["hit_top20"] else "miss"
        rank_str = f"ranked #{scored['actual_rank']}" if scored["actual_rank"] else "not in top-20"
        print(f"  Last draw ({scored['predicted_draw_date']} {scored['draw_label']}): "
              f"actual={actual}  consensus={hit_c}  top20={hit_t}  ({rank_str})")
    else:
        print("  No pending prediction to score.")
    print()
    print("  Running tally:")
    print_stats()

    # ------------------------------------------------------------------
    # Step 3: Run prediction
    # ------------------------------------------------------------------
    print()
    print("=" * 50)
    print("Step 3: Running prediction")
    print("=" * 50)

    buf = io.StringIO()
    from predict_now import run_prediction, print_results
    result = run_prediction()

    with redirect_stdout(buf):
        print_results(result)
    prediction_text = buf.getvalue()
    print(prediction_text, end="")

    # ------------------------------------------------------------------
    # Step 4: Save prediction to ledger
    # ------------------------------------------------------------------
    from prediction_ledger import save_prediction
    save_prediction(result)
    print(f"  Prediction saved to ledger.")

    # ------------------------------------------------------------------
    # Step 5: Generate analysis (Bonsai)
    # ------------------------------------------------------------------
    analysis_text = ""
    if not no_bonsai:
        print()
        print("=" * 50)
        print("Step 5: Generating analysis (Bonsai)")
        print("=" * 50)
        try:
            from analysis import generate_analysis, evaluate_prediction
            top20          = [f"{result['sorted_combos'][i]:03d}" for i in range(20)]
            consensus_strs = [f"{c:03d}" for c in result["consensus_combos"]]

            print("  Generating writeup...")
            analysis_text = generate_analysis(
                draw_label    = f"{result['draw_day']} {result['draw_label']}",
                draw_display  = result["draw_display"],
                digit_summary = result["digit_summary"],
                top_plays     = [f"{result['sorted_combos'][i]:03d}" for i in range(10)],
                best_k        = len(result["consensus_combos"]),
                best_ev       = result["consensus_ev"],
            )
            print()
            print("  [Analysis]")
            print(analysis_text)

            print()
            print("  Generating evaluation...")
            eval_text = evaluate_prediction(
                draw_label       = f"{result['draw_day']} {result['draw_label']}",
                digit_summary    = result["digit_summary"],
                consensus_combos = consensus_strs,
                consensus_ev     = result["consensus_ev"],
                n_consensus      = len(result["consensus_combos"]),
                top20_combos     = top20,
            )
            print()
            print("  [Evaluation]")
            print(eval_text)
            print()
            analysis_text = f"{analysis_text}\n\n[Evaluation]\n{eval_text}"
        except Exception as e:
            print(f"  Analysis failed (skipping): {e}")

    # ------------------------------------------------------------------
    # Step 6: Send to Telegram
    # ------------------------------------------------------------------
    if no_send:
        print()
        print("(--no-send: skipping Telegram)")
        return

    print()
    print("=" * 50)
    print("Step 6: Sending to Telegram")
    print("=" * 50)

    from telegram_bot import send_message

    header = (f"{result['draw_date'].strftime('%B %-d %Y')} — "
              f"{result['draw_day']} {result['draw_label']} ({result['draw_display']})")

    msg_lines = [f"LottAI — {header}", ""]

    # Last result recap
    if scored:
        hit_c = "HIT" if scored["hit_consensus"] else "miss"
        msg_lines.append(f"Last draw: {scored['actual_combo']}  consensus={hit_c}")
        msg_lines.append("")

    # Consensus plays
    consensus = result["consensus_combos"]
    pm        = result["pm"]
    if consensus:
        msg_lines.append(f"Consensus ({len(consensus)} plays)  EV ${result['consensus_ev']:+.3f}")
        msg_lines.append("")
        for i, combo in enumerate(consensus, 1):
            d1 = combo // 100; d2 = (combo // 10) % 10; d3 = combo % 10
            msg_lines.append(f"  {i:>2}.  {combo:03d}  ({d1}-{d2}-{d3})  p={pm[combo]:.5f}")
    else:
        msg_lines.append("No consensus — models disagree. Consider skipping this draw.")

    if analysis_text:
        msg_lines.append("")
        msg_lines.append(analysis_text)

    send_message("\n".join(msg_lines))
    print("Sent.")


if __name__ == "__main__":
    main()

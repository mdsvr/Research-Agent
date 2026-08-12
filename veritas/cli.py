import os
import sys

import click

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass  # non-reconfigurable stream (piped/redirected); default encoding is fine

from veritas.config import load_config


@click.group()
def main():
    """Veritas Agent — verified-citation research agent over a fixed corpus."""


@main.command()
@click.option("--config", default="config.yaml", help="Path to config file")
def ingest(config):
    """Ingest source corpus documents and build search indices."""
    from veritas.ingest import run_ingest
    cfg = load_config(config)
    click.echo(f"Starting ingestion for corpus: {cfg.corpus_dir}...")
    chunk_count, doc_count = run_ingest(cfg)
    click.echo(f"Ingestion complete: Loaded {doc_count} documents into {chunk_count} chunks.")


@main.command()
@click.argument("query")
@click.option("--offline", is_flag=True, help="Skip API providers; replay cache or extract")
@click.option("--config", default="config.yaml", help="Path to config file")
def ask(query, offline, config):
    """Query the agent with a question over the corpus."""
    from veritas.pipeline import run_pipeline
    cfg = load_config(config)
    click.echo(f"Querying Veritas Agent (offline={offline}): '{query}'\n")
    result = run_pipeline(query, cfg, offline=offline)

    if result.abstained:
        click.secho("[ABSTAINED]", fg="yellow", bold=True)
        click.echo(result.abstain_reason)
        return

    answer = result.final_answer
    click.secho("[ANSWER]", fg="green", bold=True)
    scores = {v.claim_text: v for v in result.verdicts}
    for claim in answer.claims:
        cites = ", ".join(claim.citations) if claim.citations else "no citations"
        verdict = scores.get(claim.text)
        suffix = f" (entailment {verdict.score:.2f} via {verdict.backend})" if verdict else ""
        click.echo(f"- {claim.text} [{cites}]{suffix}")

    if result.ungrounded_quotes:
        click.secho(f"\n{len(result.ungrounded_quotes)} claim(s) dropped — quoted text is not "
                    f"in the cited source:", fg="red")
        for quote in result.ungrounded_quotes:
            click.echo(f"  - \"{quote[:120]}\"")

    if result.dropped_claims:
        click.secho(f"\n{len(result.dropped_claims)} claim(s) dropped as unverified:", fg="yellow")
        for text in result.dropped_claims:
            click.echo(f"  - {text}")

    if answer.generator == "extractive-fallback":
        click.secho(
            "\nNote: no language model was available, so this is a verbatim extract of the "
            "top-ranked passage, not a generated answer.", fg="yellow")


@main.command(name="eval")
@click.option("--gold", default="eval/gold.jsonl", help="Path to gold eval jsonl")
@click.option("--offline", is_flag=True, help="Run evaluation without API providers")
@click.option("--ablations/--no-ablations", default=True, help="Run the full ablation ladder")
@click.option("--config", default="config.yaml", help="Path to config file")
def eval_cmd(gold, offline, ablations, config):
    """Run the evaluation suite over the gold dataset."""
    from eval.run_eval import execute_eval
    cfg = load_config(config)
    execute_eval(gold, cfg, offline=offline, ablations=ablations)


@main.command()
@click.option("--config", default="config.yaml", help="Path to config file")
def providers(config):
    """Check which generation providers are configured and reachable."""
    import requests
    from veritas.generate import build_generation_prompt, _PROVIDERS
    from veritas.schemas import Chunk

    cfg = load_config(config)
    probe = [(Chunk(chunk_id="S0::c00", doc_id="S0", doc_title="Probe",
                    text="The sky is blue.", char_start=0, char_end=16, sent_range=(0, 0)), 1.0)]
    prompt = build_generation_prompt("What colour is the sky?", probe)

    for name in cfg.llm.providers:
        call = _PROVIDERS.get(name)
        if call is None:
            click.echo(f"  {name:<12} skipped (not a live provider)")
            continue
        try:
            answer = call(prompt, cfg, {"S0::c00"})
        except requests.HTTPError as e:
            click.secho(f"  {name:<12} ERROR  HTTP {e.response.status_code} — "
                        f"{e.response.text[:120]}", fg="red")
            continue
        except requests.ConnectionError:
            click.echo(f"  {name:<12} not running — skipped")
            continue
        except Exception as e:
            click.secho(f"  {name:<12} ERROR  {type(e).__name__}: {e}", fg="red")
            continue
        if answer is None:
            click.echo(f"  {name:<12} no API key set — skipped")
        else:
            click.secho(f"  {name:<12} OK     responded as {answer.generator}", fg="green")

    click.echo("\nSet a key and re-run, e.g.:  $env:GROQ_API_KEY=\"gsk_...\"   (PowerShell)\n"
               "Also honoured: GEMINI_API_KEY, OPENROUTER_API_KEY. See .env.example.")


@main.command()
@click.option("--score", "do_score", is_flag=True, help="Read filled-in grades back")
@click.option("--grades", default="eval/human_grades.jsonl", help="Path to the grading sheet")
@click.option("--results", default="eval/results/ablation_results.json",
              help="Evaluation run to grade (use ablation_results_holdout.json for the holdout set)")
def grade(do_score, grades, results):
    """Hand-grade answer correctness, and check the lexical proxy against those grades."""
    from eval.grade import emit, score as score_grades
    score_grades(grades) if do_score else emit(results_path=results, out_path=grades)


@main.command()
@click.option("--gold", default="eval/gold.jsonl", help="Path to gold eval jsonl")
@click.option("--config", default="config.yaml", help="Path to config file")
def calibrate(gold, config):
    """Sweep the Gate A threshold against the gold set."""
    from eval.calibrate import run_calibration
    run_calibration(load_config(config), gold)


if __name__ == "__main__":
    main()

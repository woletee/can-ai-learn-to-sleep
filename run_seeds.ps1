# Run the GPT-2 sleep experiment for extra seeds, one after another (seed 0 lives in results/gpt2-full).
$d = $PSScriptRoot
$env:SLEEP_CACHE = "1"
$env:MPLBACKEND = "Agg"
$env:PYTHONUNBUFFERED = "1"
Remove-Item Env:SLEEP_ONLY -ErrorAction SilentlyContinue
foreach ($s in 1, 2) {
    $env:SLEEP_SEED = "$s"
    & "$d\.venv\Scripts\python.exe" -X utf8 "$d\sleep_experiment.py" > "$d\gpt2_seed$s.log" 2> "$d\gpt2_seed$s.err"
}
"all seeds done" | Out-File "$d\seeds_done.txt"

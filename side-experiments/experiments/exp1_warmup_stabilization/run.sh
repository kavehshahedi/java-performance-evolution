#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PART1_DIR="$PROJECT_ROOT/results/exp1/part1"
PART2_DIR="$PROJECT_ROOT/results/exp1/part2"
TRACES_DIR="$PART2_DIR/traces"
BENCHMARKS_JAR="$PROJECT_ROOT/target/benchmarks.jar"
JIB_JAR="$PROJECT_ROOT/jib.jar"
BH_OPT="-Djmh.blackhole.autoDetect=false"

mkdir -p "$PART1_DIR" "$TRACES_DIR"

if [ ! -f "$BENCHMARKS_JAR" ]; then
    echo "benchmarks.jar not found. Building project..."
    mvn -f "$PROJECT_ROOT/pom.xml" clean package -q
fi

echo "=== Experiment 1, Part A: JMH-level Warmup Stabilization ==="
echo ""

echo "[Part A, Run A] With warmup (wi=3, i=5, f=3)..."
java -jar "$BENCHMARKS_JAR" \
    "benchmarkOptimizedMethod" \
    -f 3 -wi 3 -i 5 \
    -rf json -rff "$PART1_DIR/with_warmup.json"

echo ""
echo "[Part A, Run B] Without warmup (wi=0, i=5, f=3)..."
java -jar "$BENCHMARKS_JAR" \
    "benchmarkOptimizedMethod" \
    -f 3 -wi 0 -i 5 \
    -rf json -rff "$PART1_DIR/no_warmup.json"

echo ""
echo "=== Experiment 1, Part B: Trace-based IQR Filtering ==="
echo ""

if [ ! -f "$JIB_JAR" ]; then
    echo "ERROR: jib.jar not found at $JIB_JAR — skipping Part B."
    exit 1
fi

# Remove stale trace files from any previous run
rm -f "$TRACES_DIR"/baseline_trace_*.log \
       "$TRACES_DIR"/baseline_trace_*.json \
       "$TRACES_DIR"/nowarmup_trace_*.log \
       "$TRACES_DIR"/nowarmup_trace_*.json

EXP1B_CFG_A="$SCRIPT_DIR/config_part2_baseline.yaml"
cat > "$EXP1B_CFG_A" << EOF
logging:
  file: $TRACES_DIR/baseline_trace.log
  addTimestampToFileNames: true
  useHash: true

instrumentation:
  targetPackage: com.example
  targetMethods:
    instrument:
      - public java.lang.String com.example.PerformanceComparison.optimizedMethod(int)
EOF

echo "[Part B, Run A] Agent + warmup baseline (wi=3, i=5, f=3)..."
java $BH_OPT -javaagent:"$JIB_JAR=config=$EXP1B_CFG_A" -jar "$BENCHMARKS_JAR" \
    "benchmarkOptimizedMethod" \
    -f 3 -wi 3 -i 5 \
    -rf json -rff "$PART2_DIR/with_warmup.json"

EXP1B_CFG_B="$SCRIPT_DIR/config_part2_nowarmup.yaml"
cat > "$EXP1B_CFG_B" << EOF
logging:
  file: $TRACES_DIR/nowarmup_trace.log
  addTimestampToFileNames: true
  useHash: true

instrumentation:
  targetPackage: com.example
  targetMethods:
    instrument:
      - public java.lang.String com.example.PerformanceComparison.optimizedMethod(int)
EOF

echo ""
echo "[Part B, Run B] Agent + no warmup (wi=0, i=5, f=3) + trace collection..."
java $BH_OPT -javaagent:"$JIB_JAR=config=$EXP1B_CFG_B" -jar "$BENCHMARKS_JAR" \
    "benchmarkOptimizedMethod" \
    -f 3 -wi 0 -i 5 \
    -rf json -rff "$PART2_DIR/no_warmup.json"

echo ""
echo "Results saved to $PART1_DIR/ and $PART2_DIR/"
echo "Run 'python3 $SCRIPT_DIR/analyze.py' to analyze."

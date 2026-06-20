
#!/usr/bin/env bash

set -euo pipefail



# 用法：

# ./save_version.sh 标签 t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 "备注"

#

# 时间单位统一为微秒。

#

# 示例：

# ./save_version.sh v001_all_accepted \

#   1 8 8 19 57 185 357 363 703 1395 \

#   "10个测试点全部Accepted"



if [ "$#" -lt 11 ]; then

    echo "用法："

    echo "$0 标签 t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 [备注]"

    exit 1

fi



ROOT="$(cd "$(dirname "$0")" && pwd)"

SOURCE="$ROOT/src/kernel.cu"



if [ ! -f "$SOURCE" ]; then

    echo "找不到提交文件：$SOURCE"

    exit 1

fi



LABEL="$1"

TIMES=("${@:2:10}")

NOTE="${12:-无}"



SAFE_LABEL="$(echo "$LABEL" | tr ' /:' '___' | tr -cd 'A-Za-z0-9_.-')"

STAMP="$(date '+%Y%m%d_%H%M%S')"

VERSION_DIR="$ROOT/versions/${STAMP}_${SAFE_LABEL}"



mkdir -p "$VERSION_DIR"



# 保存源代码

cp "$SOURCE" "$VERSION_DIR/kernel.cu"



# 保存哈希，确认不同版本是否确实不同

sha256sum "$VERSION_DIR/kernel.cu" > "$VERSION_DIR/sha256.txt"



# 记录 Git 信息

GIT_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo '未使用Git')"

GIT_STATUS="$(git -C "$ROOT" status --short 2>/dev/null || true)"



# 题面给出的 baseline，单位为微秒

# 测试点 8 没有给出 baseline，用空字符串表示

BASELINES=(8 1165 1170 1179 4551 9000 18000 "" 35000 67000)



# 保存逐测试点结果

echo "testcase,status,time_us,baseline_us,speedup" \

    > "$VERSION_DIR/results.csv"



for i in $(seq 0 9); do

    CASE_ID=$((i + 1))

    TIME="${TIMES[$i]}"

    BASELINE="${BASELINES[$i]}"



    if [ -n "$BASELINE" ]; then

        SPEEDUP="$(awk -v b="$BASELINE" -v t="$TIME" \

            'BEGIN { if (t > 0) printf "%.4f", b / t; else print "N/A" }')"

    else

        SPEEDUP="N/A"

    fi



    echo "$CASE_ID,Accepted,$TIME,$BASELINE,$SPEEDUP" \

        >> "$VERSION_DIR/results.csv"

done



# 保存说明

cat > "$VERSION_DIR/metadata.md" <<META

# Kernel 版本记录



- 标签：$LABEL

- 保存时间：$(date '+%Y-%m-%d %H:%M:%S')

- 源文件：src/kernel.cu

- Git Commit：$GIT_COMMIT

- 备注：$NOTE



## OJ 结果



| 测试点 | 状态 | 时间 |

|---:|---|---:|

| 1 | Accepted | ${TIMES[0]} μs |

| 2 | Accepted | ${TIMES[1]} μs |

| 3 | Accepted | ${TIMES[2]} μs |

| 4 | Accepted | ${TIMES[3]} μs |

| 5 | Accepted | ${TIMES[4]} μs |

| 6 | Accepted | ${TIMES[5]} μs |

| 7 | Accepted | ${TIMES[6]} μs |

| 8 | Accepted | ${TIMES[7]} μs |

| 9 | Accepted | ${TIMES[8]} μs |

| 10 | Accepted | ${TIMES[9]} μs |



## 重点测试点



- 测试点 5：${TIMES[4]} μs

- 测试点 7：${TIMES[6]} μs

- 测试点 10：${TIMES[9]} μs



## 保存时的 Git 工作区状态



\`\`\`text

$GIT_STATUS

\`\`\`

META



# 创建或更新总索引

INDEX="$ROOT/versions/index.csv"



if [ ! -f "$INDEX" ]; then

    echo "timestamp,label,t1_us,t2_us,t3_us,t4_us,t5_us,t6_us,t7_us,t8_us,t9_us,t10_us,note,directory" \

        > "$INDEX"

fi



ESCAPED_NOTE="$(echo "$NOTE" | tr ',' ';' | tr '\n' ' ')"



echo "$STAMP,$LABEL,${TIMES[0]},${TIMES[1]},${TIMES[2]},${TIMES[3]},${TIMES[4]},${TIMES[5]},${TIMES[6]},${TIMES[7]},${TIMES[8]},${TIMES[9]},$ESCAPED_NOTE,$VERSION_DIR" \

    >> "$INDEX"



echo

echo "版本已保存："

echo "$VERSION_DIR"

echo

echo "结果文件："

echo "$VERSION_DIR/results.csv"

echo "$VERSION_DIR/metadata.md"

echo "$VERSION_DIR/sha256.txt"


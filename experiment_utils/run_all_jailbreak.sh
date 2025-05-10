#!/bin/bash

# 依次运行 1-6 的方法
for i in {1..6}; do
    echo "运行方法 $i ..."
    python audio_jailbreak_experiment.py <<EOF
$i
EOF
done

echo "✅ 所有实验运行完毕！"

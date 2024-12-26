python -m examples.craftax.craftax_plr \
    --project jaxued-dev \
    --use_accel \
    --mode train \
    --accel_mutation claude_35_easy_hard \
    --run_name baseline-rnn-claude_35_easy_hard \
    --num_edits 1 \
    --group_name test-rnn --eval_freq 1 \

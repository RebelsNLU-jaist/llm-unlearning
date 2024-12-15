for SEED in 42
do
    for COEFFS in 1
    do
        for ALPHA in 1200
        do
            for ID in 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
            do
                for BATCH in 500
                do
                    python -m baselines.adap_rmu.unlearn \
                    --model_name_or_path "HuggingFaceH4/zephyr-7b-beta" \
                    --max_num_batches $BATCH \
                    --alpha "${ALPHA},${ALPHA}" \
                    --batch_size 4 \
                    --steering_coeffs "${COEFFS},${COEFFS}" \
                    --seed $SEED \
                    --scale 5.0 \
                    --layer_id $ID \
                    --layer_ids "$((ID - 2)),$((ID - 1)),$ID" \
                    --verbose;
                done
            done
        done
    done
done

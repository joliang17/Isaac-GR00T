# huggingface-cli download \
#     --repo-type dataset IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot \
#     --local-dir /fs/nexus-scratch/yliang17/Research/VLA/LIBERO_data

# cp examples/Libero/modality.json /fs/nexus-scratch/yliang17/Research/VLA/LIBERO_10_lerobot/meta/modality.json

huggingface-cli download \
    --repo-type dataset physical-intelligence/libero \
    --local-dir /fs/nexus-scratch/yliang17/Research/cache/libero_openpi/
export HF_ENDPOINT=https://hf-mirror.com
cd src
# Execute applications in parallel
export CUDA_VISIBLE_DEVICES="0,1,2,3"
 for count in {1..3}
 do
     python trainer.py --type 'abstract_base' --label_nums 2 --count $count --LLM gpt
     python trainer.py --type 'abstract_base' --label_nums 2 --count $count --LLM gemini
     python trainer.py --type 'abstract_base' --label_nums 2 --count $count --LLM claude
     python trainer.py --type 'abstract_base' --label_nums 2 --count $count --LLM mix
     python trainer.py --type 'hybrid' --label_nums 2 --count $count --LLM gpt
     python trainer.py --type 'hybrid' --label_nums 2 --count $count --LLM gemini
     python trainer.py --type 'hybrid' --label_nums 2 --count $count --LLM claude
     python trainer.py --type 'hybrid' --label_nums 2 --count $count --LLM mix
     python trainer.py --type 'meta_review' --label_nums 2 --count $count --LLM gpt
     python trainer.py --type 'meta_review' --label_nums 2 --count $count --LLM gemini
     python trainer.py --type 'meta_review' --label_nums 2 --count $count --LLM claude
     python trainer.py --type 'meta_review' --label_nums 2 --count $count --LLM mix
 done


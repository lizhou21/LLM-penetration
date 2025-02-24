export HF_ENDPOINT=https://hf-mirror.com
cd src
# Execute applications in parallel
export CUDA_VISIBLE_DEVICES="0"
for count in {1..3}
do
    python tester.py --count $count --LLM gpt
    python tester.py --count $count --LLM gemini
    python tester.py --count $count --LLM claude
    python tester.py --count $count --LLM mix
done

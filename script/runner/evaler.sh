export HF_ENDPOINT=https://hf-mirror.com
cd src
# Execute applications in parallel
export CUDA_VISIBLE_DEVICES="0"
 for count in {1..3}
 do
      python evaler.py --count $count --journal iclr --type abstract
      python evaler.py --count $count --journal iclr --type meta_review
      python evaler.py --count $count --journal iclr --type review
     python evaler.py --count $count --journal acl
     python evaler.py --count $count --journal cvpr
     python evaler.py --count $count --journal emnlp
     python evaler.py --count $count --journal icml
     python evaler.py --count $count --journal ijcai
     python evaler.py --count $count --journal neurips
 done

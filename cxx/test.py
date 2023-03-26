import geneasm
import geneasm.native
import time

t0 = time.monotonic()

asmConfig = geneasm.native.AssemblerConfig()
asmConfig.max_ntds_per_edge_ = geneasm.CONFIG.dict['max_ntds_per_edge']
asmConfig.kmer_len = geneasm.CONFIG.dict['kmer_len']
asmConfig.alphabet_len = geneasm.CONFIG.dict['alphabet_len']
asmConfig.work_queue_low_watermark = geneasm.CONFIG.dict['work_queue_low_watermark']
asmConfig.work_queue_high_watermark = geneasm.CONFIG.dict['work_queue_high_watermark']
asmConfig.knn_top_k = geneasm.CONFIG.dict['knn_top_k']

print(asmConfig)

asm = geneasm.native.Assembler(\
	"../scratch/simsample.fasta",
	geneasm.native.kGeneSeqTypeDNA,
	asmConfig)

#geneasm.native.run_unit_tests()
elapsed_wallclock_time = time.monotonic() - t0

print(elapsed_wallclock_time)
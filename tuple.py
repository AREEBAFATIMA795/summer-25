#tuple
gene_1 = ((100,200),(500,600))
for start , end in gene_1:
    print("the position of gene_1 is start from:",start,"the ending of this position:",end)

bases = set('ATGCGGTCCTGC')
print("the unique nucleotide in bases tuple is:",bases)
##############################################################################


gene_dic = {'brca1':1,'tp53':2}
print(type(gene_dic))
print(gene_dic)
for name , num in gene_dic.items():
    print("the name of gene is:",name,"and the number in this:",num)
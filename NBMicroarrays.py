from NBHighThroughput import *

class NBMicroarrays(HighThroughput):

    def read_exprs_data(self):
        """
        Reads Microarrays expression data file.

        """
        print("Reading Microarrays gene expression file...")
        exprs = pd.read_table(self.ht_file, header=0, sep='\t')
        exprs_1 = exprs.filter(regex="NB").transpose()
        exprs_1.columns = exprs['GENE']
        self.exprs = exprs_1
        self.exprs = self.exprs.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        print("Expression data successfully load.")










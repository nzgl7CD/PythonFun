import pandas as pd
import timeit
class Solution:
    def __init__(self) -> None:
        self.data=self.get_data()
    def get_data(data):
        ds = pd.read_excel('private equity\dfBRP.xlsx', sheet_name='data')
        return ds.iloc[:, 0].dropna()
    def without_our_tech(self):
        filtered_data = []
        for name in self.data:
            if name.startswith('A'):  
                filtered_data.append(name.upper())  
        result_df = pd.DataFrame(filtered_data, columns=['CompanyName'])
        result_df.to_excel('private equity/without_our_tech.xlsx', index=False)
    def with_our_tech(self):
        filtered_data = self.data[self.data.str.startswith('A')].str.upper()
        result_df = pd.DataFrame(filtered_data, columns=['CompanyName'])
        result_df.to_excel('private equity/with_our_tech.xlsx', index=False)
    def measure_time(self):
        without_tech_time = timeit.timeit(self.without_our_tech, number=1)
        print("\nData flow without our tech Execution Time: {:,.6f} seconds".format(without_tech_time), end='\n'*2)
        with_tech_time = timeit.timeit(self.with_our_tech, number=1)
        print("Data flow with our tech Execution Time: {:,.6f} seconds".format(with_tech_time), end='\n'*2)
        print("Our tech is {0:,.3f} times faster than our competitors".format(without_tech_time/with_tech_time), end='\n'*2)
solution = Solution()
solution.measure_time()
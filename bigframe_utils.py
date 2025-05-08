import bigframes.pandas as bpd
from bigframes.ml.llm import GeminiTextGenerator

class mock_data_generator:

    def __init__(self, proj, dataset, table, batch_size):

        bpd.options.bigquery.project = proj
        self.model = GeminiTextGenerator(model_name='gemini-2.0-flash-001')
        self.dataset = dataset
        self.table = table
        self.batch_size = batch_size
        self.llm_code = None
        self.cloud_llm_code_executor = None

    def generate_prompt(self, perc_table_sample, n_prompt_max_records, max_tries):

        print('Generating prompt')

        df = bpd.read_gbq(f'FROM {self.dataset}.{self.table} TABLESAMPLE SYSTEM ({perc_table_sample} PERCENT)')

        prompt = f"""Write Python code to generate a pandas DataFrame mimicking
        the characteristics of an existing DataFrame. The schema of the output
        dataframe has to match the schema of the input DataFrame exactly.

        The characteristics of the existing DataFrame are as follows:
        {str(df.describe())}

        Sample data from the existing DataFrame:
        {str(df.sample(n_prompt_max_records))}

        Note:
          - Return the code only, no additional texts or comments
          - Use the 'faker' library to generate the contents of the DataFrame
          - Generate {self.batch_size} rows
          - Call the output DataFrame 'df_mock_data'
        """

        df_prompt = bpd.DataFrame({'prompt': [prompt]})

        llm_code = None
        for i in range(max_tries):
            llm_code = self.model.predict(df_prompt)['ml_generate_text_llm_result'].iloc[0][9:-3]
            try:
                exec(llm_code, globals())
                break
            except Exception:
                if i == max_tries - 1:
                    raise Exception('LLM failed to generate executable code; terminating')
                else:
                    print('LLM failed to generate executable code; retrying')

        self.llm_code = llm_code

        print('Prompt successfully generated')

    def deploy_remote_function(self):

        print('Deploying remote function')

        @bpd.remote_function(
            input_types=[str],
            output_type=str,
            reuse=True,
            name='exp_python_code_executor',
            packages=['faker', 'pandas'],
            cloud_function_service_account='default'
        )
        def executor(code):
            context = {}
            exec(code, context)
            return context.get('df_mock_data').to_json(orient='records')

        self.cloud_code_executor = executor

        print('Remote function successfully deployed')

    def run_remote_function(self, n_mock_records):

        print('Executing LLM-generated code via remote function')

        df = bpd.DataFrame(self.llm_code, range(int(n_mock_records / self.batch_size)), columns=['mock_data'])
        df['mock_data'] = df['mock_data'].apply(self.cloud_code_executor)

        print('LLM-generated code successfully executed')

        return df

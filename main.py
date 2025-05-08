from bigframe_utils import mock_data_generator

if __name__ == '__main__':

    # Change my debug arguments to your own resources before running this...

    generator = mock_data_generator(
        proj='sandbox-kkenesei',
        dataset='dummy_us',
        table='taxi_trips_chicago',
        batch_size=1000
    )

    generator.generate_prompt(perc_table_sample=.1, n_prompt_max_records=50, max_tries=5)

    #print(generator.llm_code)

    generator.deploy_remote_function()

    mock_data = generator.run_remote_function(n_mock_records=5000)

    print(mock_data.sample(10))

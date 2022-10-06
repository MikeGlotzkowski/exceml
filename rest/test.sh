curl --location --request GET 'http://127.0.0.1:5000/create_experiment'

curl --location --request POST 'http://127.0.0.1:5000/upload_csv_for_experiment' \
--form 'file=@"/home/sebastian/dev/exceml/uploads/classification_example.csv"' \
--form 'id="8"' \
--form 'y_column="y"'


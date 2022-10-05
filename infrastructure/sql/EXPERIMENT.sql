CREATE TABLE experiment (
            id SERIAL PRIMARY KEY,
            x_df BYTEA,
            y_df BYTEA,
            y_column VARCHAR(80),
            x_columns BYTEA,
            columns BYTEA,
            results BYTEA,
            best BYTEA
        )
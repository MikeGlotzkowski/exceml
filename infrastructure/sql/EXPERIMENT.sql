CREATE TABLE IF NOT EXISTS experiments (
        id SERIAL PRIMARY KEY,
        x_df BYTEA,
        y_df BYTEA,
        y_column VARCHAR,
        x_columns BYTEA,
        columns BYTEA,
        results BYTEA,
        best BYTEA,
        sprint_statistics BYTEA,
        leaderboard BYTEA
    );
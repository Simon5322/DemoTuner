CREATE TABLE usertable (
    YCSB_KEY VARCHAR(255) PRIMARY KEY
);

DO $$
DECLARE
    i INT := 0;
BEGIN
    FOR i IN 0..(fieldcount-1) LOOP
        EXECUTE 'ALTER TABLE usertable ADD COLUMN FIELD' || i || ' TEXT';
    END LOOP;
END $$;

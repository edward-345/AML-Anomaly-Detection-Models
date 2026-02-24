--------------------------------------------------------------------
-- For Version A
--------------------------------------------------------------------

-- INDIVIDUALS ONLY FROM VERSION A
CREATE TABLE individuals_vA AS
SELECT * FROM Kmeans_Seperate ks
WHERE cluster LIKE 'individual%'
ORDER BY cluster ASC;

SELECT * FROM individuals_vA;

-- JOIN WITH VersionA_Individual_BeforeFAMD
-- THIS ONE IS RAW INOUT OF ROWS THAT ARE IN A CLUSTER 
CREATE VIEW indVA_clusters AS
SELECT
a.*,
i.cluster
FROM VersionA_Individual_BeforeFAMD a
INNER JOIN individuals_vA i
ON a.customer_id = i.customer_id;

SELECT * FROM indVA_clusters;

-- RAW INPUT OF ROWS IN CLUSTER 0
CREATE TABLE ind0_vA_ AS
SELECT * FROM indVA_clusters
WHERE cluster = "individual_0";

-- RAW INPUT OF ROWS IN CLUSTER 1
CREATE TABLE ind1_vA AS 
SELECT * FROM indVA_clusters
WHERE cluster = "individual_1";

-- RAW INPUT OF ROWS IN CLUSTER 2
CREATE TABLE ind2_vA AS 
SELECT * FROM indVA_clusters
WHERE cluster = "individual_2";

-- RAW INPUT OF ROWS IN CLUSTER 3
CREATE TABLE ind3_vA AS 
SELECT * FROM indVA_clusters
WHERE cluster = "individual_3";

-- RAW INPUT OF ROWS IN CLUSTER 4
CREATE TABLE ind4_vA AS 
SELECT * FROM indVA_clusters
WHERE cluster = "individual_4";

-- RAW INPUT OF ROWS IN CLUSTER 5
CREATE TABLE ind5_vA AS 
SELECT * FROM indVA_clusters
WHERE cluster = "individual_5";

-- RAW INPUT OF ROWS IN CLUSTER 6
CREATE TABLE ind6_vA AS 
SELECT * FROM indVA_clusters
WHERE cluster = "individual_6";

-- RAW INPUT OF ROWS IN CLUSTER 7
CREATE TABLE ind7_vA AS 
SELECT * FROM indVA_clusters
WHERE cluster = "individual_7";

-- RAW INPUT OF ROWS IN CLUSTER 8
CREATE TABLE ind8_vA AS 
SELECT * FROM indVA_clusters
WHERE cluster = "individual_8";

SELECT 'ind0_vA' AS table_name, COUNT(*) AS row_count FROM ind0_vA_
UNION ALL
SELECT 'ind1_vA', COUNT(*) FROM ind1_vA
UNION ALL
SELECT 'ind2_vA', COUNT(*) FROM ind2_vA
UNION ALL
SELECT 'ind3_vA', COUNT(*) FROM ind3_vA
UNION ALL
SELECT 'ind4_vA', COUNT(*) FROM ind4_vA
UNION ALL
SELECT 'ind5_vA', COUNT(*) FROM ind5_vA
UNION ALL
SELECT 'ind6_vA', COUNT(*) FROM ind6_vA
UNION ALL
SELECT 'ind7_vA', COUNT(*) FROM ind7_vA
UNION ALL
SELECT 'ind8_vA', COUNT(*) FROM ind8_vA
ORDER BY table_name;

SELECT COUNT(*) FROM individuals_vA;

-- SO NOW FOR EACH VERSION A CLUSTER FOR INDVIDUALS WE HAVE A TABLE OF THEIR RAW INPUT INFO
SELECT *
FROM ind8_vA
WHERE customer_id IN (
    'SYNID0100957188',
    'SYNID0101421130',
    'SYNID0105593361',
    'SYNID0107334515',
    'SYNID0107464935',
    'SYNID0107832828',
    'SYNID0200187014',
    'SYNID0200496670',
    'SYNID0200755574',
    'SYNID0200755995',
    'SYNID0200441116',
    'SYNID0103912349',
    'SYNID0108560369',
    'SYNID0109015075'
);


SELECT
i.*,
l.customer_id
FROM ind8_LOF_top400 l
INNER JOIN indVA_clusters i
ON l.customer_id = i.customer_id;
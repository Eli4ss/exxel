#import function 
#from databasefunction import *
from YourDataPreprocessingModule import *
import pandas as pd
def dataset_extraction():
    #extract data from database
    query1 = '''SELECT [Expedition qty].[# Serie], [Expedition qty].[Qty envoyé], [Expedition qty].wo, [Expedition qty].Docket, [Expedition qty].Loc, [Expedition qty].BL, [Expedition qty].[HRS EXP], Serial.Items, Serial.[No Série], Serial.Lot, [Doc exp].Client, Serial.[Cost Unit], Serial.TRCostUN, Serial.rwcost, [Doc recp].[Taux dev], ([Cost Unit]*[Taux dev])+[TRCostUN]+[rwcost] AS [T COST], [T COST]*[Qty envoyé] AS [G COST], Serial.[Qty-recus], [T COST]*[Qty-recus] AS [COST R], [Doc recp].[br-clef], [Expedition qty].[expedition-clef], [WORK-ORDER].[WO-NO]
    FROM [WORK-ORDER] RIGHT JOIN ([Doc recp] INNER JOIN ([Doc exp] RIGHT JOIN (Serial INNER JOIN [Expedition qty] ON Serial.[Serial-clef] = [Expedition qty].[# Serie]) ON [Doc exp].[bl-clef] = [Expedition qty].[Doc exp]) ON [Doc recp].[br-clef] = Serial.Reception) ON [WORK-ORDER].[WO-NO] = [Doc recp].DEWO;
    '''
    df = extract_data_from_access(query1)
    #convert columns to numerical the columns 'Docket', 'wo', 'WO-NO'
    columns = ['Docket', 'wo', 'WO-NO']
    for column in columns:
        df = convert_to_numerical(df, column)

    #extract Serial table
    query2 = '''SELECT * FROM Serial'''
    Serial = extract_data_from_access(query2)

    #rename the column '# Serie' to 'Serial-clef'
    df.rename(columns={'# Serie': 'Serial-clef'}, inplace=True)
    
    #merge the two tables
    df_merge = pd.merge(df, Serial, on='Serial-clef')

    #drop useless columns
    columns = [ 'wo', 'Loc',  'HRS EXP','Taux dev', 'No Série_y','Items_y', 'Lot_y', 'Qty-recus_y', 'TAREKG','HRS SERIAL', 'Cost Unit_y','TRCostUN_y', 'QTY-FOUR','rwcost_y', 'print ctrl', 'Client_y','hrsst', 'hrsend', 'hrsst2', 'hrsend2', 'hrs total', 'pers', 'pcused','hrs t numb', 'deviseserial', 'etape', 'SAMPLE', 'COMPTNOTE','printcompt', 'negosurprix', 'cb_pour','INFOETI', 'cost_maj', 'EMP', 'QC_PRISE', 'COSTA', 'COSTB', 'COSTC', 'COSTD', 'COSTE', 'COSTF','Client_x', 'Cost Unit_x', 'TRCostUN_x', 'rwcost_x','T COST', 'G COST', 'COST R']
    df_merge = drop(df_merge, columns)

    #rename some columns after joining
    df_merge.rename(columns={'Items_x': 'Items', 'No Série_x': 'No Série', 'Qty-recus_x': 'Qty-recus', 'Lot_x': 'Lot'}, inplace=True)
    
    return df_merge



#extract data from database

def preprocess_dataIzod():
  df = dataset_extraction()
  #construct the dataset for izod 
  IzodColumns = ['Items','melt', 'flex', 'traction','COULEUR', 'I_F', 'I_CM','I_G', 'I1', 'izode']
  df_izod = df[IzodColumns]
  #rename the columns izode to I48
  df_izod.rename(columns={'izode': 'I48'}, inplace=True)


  #convert to numeric the columns I_F, I_CM, I_G, Items if it is possible
  columns = ['I_F', 'I_CM', 'I_G', 'Items']
  for column in columns:
      df_izod = convert_to_numerical(df_izod, column)

  #replace values in the column ITEMS
  df_izod['Items'] = df_izod['Items'].replace({54: 'NORYL',
    55: 'DIVERS',
    56: 'EVOPRENE-RP',
    63: 'ELASTOLLAN-RP',
    75: 'ADDITIFS',
    102: 'PBT-RG',
    112: 'PPH',
    116: 'PVC-FLEX',
    127: 'PP-RG/RP',
    133: 'ABS/AC-C',
    134: 'LLDPE',
    163: 'PEHD-ROLLS',
    165: 'ASA',
    173: 'SYNPRENE',
    185: 'ABS/AC-BALE',
    199: 'PBT-RP',
    208: 'PP-RP20/2/5',
    214: 'TRI-WIN',
    215: 'PP-RP20/3/2',
    216: 'PP-RP2/10/2',
    225: 'ABS/AC-RG',
    232: 'ABS',
    233: 'ACETAL',
    234: 'AC',
    235: 'CACO3',
    236: 'CONPE',
    237: 'CONPP',
    238: 'CONPS',
    239: 'EMA',
    240: 'EVA',
    241: 'GELOY',
    242: 'PS',
    243: 'GPPS',
    244: 'MIPS',
    245: 'HIPS',
    246: 'HMW',
    247: 'META',
    248: 'NYLON',
    249: 'PC/ABS',
    250: 'PC',
    251: 'PE/PP',
    252: 'HDPE',
    253: 'LDPE',
    254: 'PETE',
    255: 'PLA',
    256: 'PP',
    257: 'PVC',
    258: 'SANTO',
    259: 'SURLYN',
    260: 'TPE',
    261: 'TPO',
    262: 'TPR',
    263: 'REJET',
    264: 'MDPE',
    265: 'PETG',
    266: 'PEROX',
    267: 'PEHD-2057C',
    268: 'CONEVA',
    269: 'PP/PE',
    270: 'KRATON PS',
    271: 'KRATON PP',
    272: 'CARTON',
    273: 'SACS',
    274: 'Exxelclean',
    275: 'TRANSPORT-EXL-528099',
    276: 'PIECES ET EQUIPEMENT',
    277: 'GAYLORDS',
    278: 'PAL CPC',
    279: 'LAVAGE',
    280: 'MANUTENTION',
    281: 'VIDANGE',
    282: 'GRANULATION',
    283: 'EMBALLAGE',
    284: 'EXXELMAX',
    285: 'DELRIN',
    286: 'ABS/PA',
    290: 'SAN',
    291: 'TPU',
    292: 'GAYLORDS-IN',
    293: 'GAYLORDS-OUT EXL-527854',
    294: 'PAL-SCRAP EXL-527853',

    296: 'PAL-CLIENT EXL-528100'})

  #replace values in the column COULEUR
  df_izod['COULEUR'] = df_izod['COULEUR'].replace({1: 'BE',
    2: 'BK',
    3: 'WH',
    4: 'NAT',
    5: 'MIX',
    6: 'BLU',
    7: 'RED',
    8: 'GRE',
    9: 'YEL',
    10: 'BRN',
    11: 'GREY',
    12: 'GREYI',
    13: 'YELI',
    14: 'CL',
    15: 'GD',
    17: 'SMK',
    18: 'BRNI'})

  #replace values in the column I_G
  df_izod['I_G'] = df_izod['I_G'].replace({1: 1,
    2: 2,
    3: 3,
    4: 'L',
    5: 'E',
    6: 'CH',
    7: 'PUF',
    8: 'T',
    9: 'EMB',
    10: 'V',
    11: 'X',
    12: 'F',
    13: 'C',
    14: 'MEL',
    15: 'FO',
    16: 'R',
    17: 'S',
    18: 'HF'})

  #replace values in the column I_F
  df_izod['I_F'] = df_izod['I_F'].replace({1: 'B',
    2: 'C',
    3: 'RG',
    4: 'RL',
    5: 'RP',
    6: 'PW',
    7: 'D',
    8: 'WS',
    9: 'G',
    10: 'PR',
    12: 'P'})

  #replace values in the column I_CM
  df_izod['I_CM'] = df_izod['I_CM'].replace({1: 'EXT', 2: 'INJ', 3: 'OFF', 4: 'X', 5: 'HM'})

  #replace the comma by a dot in the values of all attribute 
  df_izod = df_izod.replace(",", ".",regex=True)
  #delete all the spaces in the values of all attribute
  df_izod = df_izod.replace(" ", "",regex=True)
  columns = ['Items','I_F', 'I_CM','I_G', 'I1', 'I48','COULEUR']
  for col in columns:
      df_izod = dropnaand0(df_izod,col)
  return df_izod
  #drop all duplicated rows 
  df_izod = df_izod.drop_duplicates()
  #drop rows that contain missing values
  df_izod = df_izod.dropna()

  
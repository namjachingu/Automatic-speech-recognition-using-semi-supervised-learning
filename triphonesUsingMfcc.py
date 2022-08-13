from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout, BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.callbacks import Callback
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier


import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pickle
import pandas as pd
import random
import sys
import itertools
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

#Show full numpy array
np.set_printoptions(threshold=sys.maxsize)

#We have the saved pickle file, now we need to access the pickled file:
# open a file, where you stored the pickled data
file = open('mfcc_train.pckl', 'rb')

# dump information to that file
mfcc_train = pickle.load(file)
# close the file
file.close()

file = open('mfcc_test.pckl', 'rb')
mfcc_test = pickle.load(file)
file.close()

file = open('mfcc_dev.pckl', 'rb')
mfcc_val = pickle.load(file)
file.close()




states = {'sil': [0, 54, 49], 'aa': [1, 134, 234, 394, 426, 574, 592, 610, 624, 816, 817, 849, 1065, 1213, 1315, 1470, 1506, 1507, 1595, 1596, 1618, 1628, 1666, 1691, 1721, 57, 115, 283, 380, 624, 791, 903, 938, 941, 1008, 1083, 1131, 1359, 1400, 1505, 1573, 95, 165, 321, 387, 541, 682, 699, 749, 1414, 1597, 1784, 1803], 'ae': [59, 168, 296, 331, 383, 479, 606, 769, 806, 973, 976, 1201, 1335, 1426, 1456, 1737, 1800, 1830, 52, 666, 784, 841, 922, 1095, 1119, 1298, 1306, 1347, 1406, 1424, 1471, 1517, 1594, 1648, 1670, 1676, 1678, 1717, 1797, 2, 152, 198, 215, 464, 505, 716, 892, 911, 970, 1021, 1023, 1192, 1281], 'ah': [3, 143, 245, 585, 586, 884, 1040, 1154, 1211, 1288, 1316, 1459, 1532, 1539, 1571, 1779, 1782, 1815, 1826, 1865, 77, 260, 452, 570, 751, 821, 885, 1117, 1343, 1550, 1570, 1713, 1729, 1852, 1856, 166, 356, 581, 727, 851, 989, 1006, 1078, 1257, 1713, 1714], 'ao': [4, 69, 157, 225,420, 485, 523, 601, 602, 622, 638, 665, 906, 936, 1019, 1068, 1262, 1362, 1443, 1465, 1525, 1582, 1635, 1642, 1707, 1791, 1792, 167, 444, 781, 889, 1019, 1224, 1250, 1536, 1566, 1790, 100, 117, 186, 219, 358, 623, 652, 765, 824, 1546, 1754, 1789], 'aw': [612, 628, 636, 811, 1181,1331, 1616, 1814, 76, 521, 932, 1003, 1232, 1549, 5, 110, 270, 330, 812, 994, 1175, 1294, 1428, 1831, 1843],  'ax': [112, 206, 466, 562, 876, 951, 999, 1203, 1333, 1436, 1547, 1654, 1719, 1723, 1734, 1785, 1861, 1869, 6, 177, 218, 278, 291, 323, 425, 549, 683, 750, 804, 1070, 1196, 1233, 1252, 1348, 1806, 1810, 1861, 1885, 291, 336, 466, 549, 561, 654, 750, 753, 804, 860, 1096, 1097, 1144, 1233, 1364, 1464, 1527, 1806, 1817, 1854, 1880], 'ay': [7, 75, 326, 441, 609, 772, 773, 913, 955, 1012, 1035, 1243, 1275, 1313, 1346, 1444, 1542, 1560, 1602, 1606, 1643, 1644, 1645, 217, 857, 909, 1012, 1164, 1317, 1318, 1329, 1346, 1629, 1639, 50, 92, 181, 339, 355, 375, 378, 544, 546, 745, 833, 1141, 1165, 1208, 1329, 1346, 1447, 1492, 1553], 'b': [481, 626, 757, 802, 943, 1177, 1278, 98, 391, 402, 589, 801, 871, 943, 944, 1277, 8, 158, 265, 415, 584, 715, 748, 912, 943, 1066], 'ch': [9, 135, 1029, 1475, 1520, 1775, 197, 332, 545, 759, 958, 1029, 1220, 1412, 1610, 1674], 'cl': [10, 60, 179, 244, 248, 320, 351, 366, 418, 448, 467, 511, 551, 583, 607, 639, 879, 887, 977, 1045, 1058, 1079, 1110, 1182, 1189, 1319, 1388, 1395, 1423, 1457, 1460, 1544, 1556, 1637, 1682, 1686, 1733, 88, 170, 220, 250, 251, 294, 338, 365, 595, 598, 831, 858, 861, 878, 916, 931, 1004, 1044, 1088, 1176, 1241, 1370, 1373, 1386, 1515, 1524, 1540, 1608, 1636, 1673, 1684, 1686, 1701, 1736, 1751, 56, 129, 146, 169, 211, 243, 293, 308, 386, 408, 411, 414, 721, 831, 868, 1024, 1087, 1174, 1349, 1425, 1486, 1564, 1683, 1700, 1703, 1766], 'd': [11, 498, 502, 508, 1009, 1401, 1600, 1726, 205, 502, 554, 898, 918, 1325, 1420, 1452, 126, 184, 205, 501, 593, 762, 893, 894, 920, 1325, 1387], 'dh': [12, 125, 335, 518, 1067, 1129, 1149, 1296, 1304, 1320, 196, 359, 635, 760, 946, 1848, 1855, 108, 368, 785, 908, 1129, 1194, 1273, 1599, 1794, 1809], 'dx': [651, 778, 832, 1013, 1240, 1502, 1662, 149, 1114, 1476, 1702, 13, 288, 633, 1094, 1310, 1503, 1529], 'eh': [14, 164, 317, 403, 416, 461, 486, 573, 670, 741, 901, 1011, 1059, 1237, 1270, 1378, 1498, 1584, 1601, 1652, 1658, 1663, 1689, 1748, 1767, 1801, 1874, 1876, 63, 102, 162, 423, 462, 531, 687, 901, 1120, 1230, 1251, 1260, 1495, 1706, 1812, 136, 202, 207, 377, 407, 492, 631, 708, 910, 1121, 1225, 1258, 1495, 1561], 'el': [15, 547, 648, 1055, 1402, 1581, 1888, 1893, 104, 455, 718, 1461, 1485, 1842, 1879, 173, 273, 342, 815, 995, 1336, 1461, 1866, 1878], 'en': [16, 771, 983, 1763, 1894, 290, 1091, 1323, 1339, 1895, 228, 525, 657, 1053, 1836], 'epi': [17, 345, 924, 1687, 1732, 345, 364, 528, 872, 1718, 1732, 872, 877, 963, 1338, 1718, 1732, 1802], 'er': [18, 124, 472, 520, 522, 538, 577, 616, 681, 722, 810, 827, 882, 921, 1071, 1173, 1183, 1212, 1434, 1451, 1453, 1462, 1513, 1563, 1578, 1580, 1603, 1604, 1656, 1657, 1669, 51, 187, 192, 249, 341, 412, 480, 490, 565, 605, 770, 865, 888, 917, 1038, 1046, 1199, 1231, 1383, 1391, 1516, 1534, 1559, 1638, 1655, 1697, 74, 127, 145, 292, 314, 406, 491, 537, 559, 600, 742, 803, 862, 919, 925, 975, 1005, 1060, 1116, 1158, 1185, 1467, 1512, 1649], 'ey': [19, 147, 174, 333, 346, 433, 504, 823, 1010, 1082, 1268, 1301, 1321, 1382, 1523, 1626, 1795, 1805,1839, 1841, 1882, 71, 240, 445, 625, 1089, 1153, 1299, 1398, 1545, 1557, 1585, 1849, 1857, 1883, 1886, 1889, 1892, 55, 212, 247, 252, 434, 517, 603, 676, 705, 867, 899, 935, 1113, 1269, 1667, 1681, 1783, 1891], 'f': [20, 266, 376, 470, 558, 560, 863, 956, 980, 1187, 1358, 1407, 68, 107, 216, 347, 552, 557, 572, 674, 743, 1168, 254, 347, 515, 572, 744, 754, 934, 1167, 1186, 1198, 1256], 'g': [21, 474, 1478, 1479, 257, 318, 685, 907, 1169, 161, 257, 720, 826, 1477, 1572], 'hh': [396, 580, 655, 1105, 1133, 1350, 1354, 1396, 1634, 1664, 1675, 1679, 1828, 1829, 148, 155, 350, 566, 755, 880, 1289, 1538, 1625, 1832, 22, 286, 316, 859, 968, 971, 1161, 1172, 1450, 1497, 1765], 'ih': [132, 153, 263, 389, 487, 510, 691, 818, 883, 896, 959, 1034, 1061, 1126, 1244, 1380, 1390, 1484, 1494, 1499, 1537, 1609, 1617, 1646, 1647, 1709, 1730, 1770, 1845, 1859, 1860, 1871, 1875, 23, 62, 185, 284, 468, 478, 702, 792, 839, 967, 1041, 1309, 1409, 1469, 1659, 1693, 1735, 87, 180, 194, 325, 329, 582, 619, 627, 647, 768, 828, 967, 1085, 1586, 1735], 'ix': [24, 111, 235, 334, 354, 410, 442, 482, 599, 663, 700, 1150, 1267, 1324, 1371, 1558, 1612, 1641, 1712, 1725, 1727, 1771, 1781, 1808, 1824, 1825, 1837, 1840, 1870, 81, 89, 259, 281, 392, 393, 454, 659, 830, 1049, 1104, 1200, 1253, 1291, 1326, 1330, 1463, 1680, 1710, 1769, 1808, 1820, 1840, 1850, 1872, 116, 222, 281, 322, 370, 453, 550, 926, 1049, 1052, 1115, 1276, 1291, 1381, 1411, 1458, 1631, 1680, 1722, 1728, 1823], 'iy': [70, 191, 233, 303, 306, 357, 449, 512, 563, 618, 634, 783, 842, 869, 964, 996, 1190, 1202, 1218, 1254, 1342, 1369, 1431, 1432, 1437, 1449, 1472, 1493, 1543, 1757, 1758, 1772, 1774, 25, 123, 236, 237, 542, 621, 667, 713, 761, 965, 1007, 1138, 1242, 1284, 1356, 1393, 1521, 1551, 1552, 1633, 1653, 1661, 1704, 1756, 53, 103, 119, 189, 213, 241, 421, 422, 424, 646, 814, 844, 900, 942, 1015, 1036, 1074, 1122, 1138, 1145, 1229, 1368, 1372, 1551, 1715, 1716], 'jh': [26, 1226, 1367, 1755, 209, 253, 1018, 144, 344, 604, 678, 680, 1421, 1508, 1838], 'k': [27, 246, 315, 495, 794, 1016, 1101, 1227, 1340, 1341, 1468, 114, 279, 477, 579, 693, 840, 915, 1102, 1147, 1209, 1214, 1312, 1496, 79, 203, 255, 277, 328, 381, 430, 432, 519, 649, 686, 738, 793, 822, 969, 1293, 1389, 1496], 'l': [28, 67, 99, 276, 343, 382, 514, 540, 594, 669, 746, 766, 800, 836, 886, 949, 1048, 1072, 1075, 1081, 1178, 1179, 1180, 1188, 1271, 1290, 1344, 1394, 1510, 1569, 86, 122, 182, 287, 469, 553, 696, 735, 736, 789, 864, 1028, 1106, 1108, 1109, 1283, 1305, 1455, 1568, 1630, 139, 188, 227, 340, 367, 379, 597, 664, 735, 805, 837, 940, 984, 1076, 1090, 1107, 1140, 1142, 1204, 1223, 1345, 1510, 1531], 'm': [29, 101, 299, 447, 656, 672, 689, 697, 954, 960, 978, 1063, 1124, 1125, 1143, 1156, 1157, 1577, 160, 443, 483, 734, 854, 945, 978, 1025, 1063, 1084, 1086, 1171, 1473, 1535, 1567, 83, 91, 138, 256, 274, 275, 483, 484, 587, 608, 673, 731, 838, 937, 1026, 1155, 1374, 1504], 'n': [30, 66, 567, 611, 617, 694, 853, 891, 979, 1054, 1162, 1195, 1249, 1376, 1441, 1446, 1575, 1579, 1607, 1627, 1692, 1743, 1760, 1762, 105, 193, 232, 319, 360, 398, 526, 588, 752, 756, 852, 1002, 1152, 1195, 1248, 1249, 1274, 1332, 1375, 1377, 1518, 82, 106, 172, 175, 352, 372, 397, 429, 588, 591, 630, 643, 834, 835, 875, 985, 1032, 1092, 1100, 1205, 1363,1500, 1695, 1745, 1773], 'ng': [690, 717, 902, 1351, 1385, 1665, 137, 413, 719, 966, 1014, 1206, 31, 280, 300, 571, 905, 1352, 1385, 1593, 1708], 'ow': [64, 109, 363, 494, 578, 613, 662, 763, 1043, 1247, 1279, 1541, 1583, 1632, 1677, 1711, 1819, 1847, 1877, 178, 262, 458, 726, 847, 991, 1062, 1287, 1300, 1548, 1555, 1696, 1834, 32, 231, 261, 298, 698, 737, 992, 1042, 1132, 1308, 1487, 1746, 1787, 1833, 1835], 'oy': [576, 895, 1448, 1483, 1761, 130, 1148, 1864, 33, 190, 950, 1307, 1430, 1739], 'p': [34, 615, 645, 764, 796, 1454, 72, 369, 404, 417, 456, 874, 961, 1170, 1216, 1217, 1490, 151, 327, 456, 457, 493, 614, 695, 797, 1136, 1217, 1282, 1334, 1511], 'r': [35, 61, 94, 239, 374, 431, 503, 509, 724, 740, 758, 779, 807, 914, 981, 1051, 1077, 1103, 1261, 1263, 1366, 1509, 1514, 1605, 1685, 1688, 1690, 171, 258, 313, 437, 463, 497, 530, 704, 739, 790, 829, 856, 957,974, 997, 1146, 1236, 1264, 1408, 1417, 1439, 1519, 1615, 1619, 78, 141, 159, 302, 362, 399, 435, 436, 437, 497, 564, 640, 658, 733, 739, 782, 990, 1207, 1210, 1286, 1397, 1418, 1429, 1519, 1530, 1740], 's': [36, 120, 131, 230, 267, 295, 309, 440, 471, 499, 536, 660, 767, 819, 845, 933, 1057, 1093, 1127, 1303, 1311, 1314, 1365, 1403, 1501, 1660, 1699, 1804, 1851, 1858, 1890, 48, 84, 513, 534, 535, 661, 688, 788, 1303, 1528, 1750, 1881, 1887, 58, 118, 183, 268, 348, 373, 388, 653, 730, 799, 846, 962, 1000, 1135, 1184, 1360, 1474, 1528, 1587, 1747, 1844, 1884], 'sh': [37, 142, 460, 820, 1137, 1238, 1239, 1399, 80, 264, 728, 1415, 1416, 1438, 80, 337, 516, 642, 1001, 1022, 1037, 1694], 't': [38, 385, 529, 947, 1080, 1197, 1228, 1410, 1590, 1788, 121, 272, 507, 590, 684, 701, 1056, 1099, 1404, 1533, 1788, 90, 204, 271, 312, 427, 446, 465, 506, 709, 848, 850, 1017, 1215, 1788], 'th': [39, 787, 1123, 1873, 361, 873, 927, 214, 361, 777, 1031, 1255, 1413], 'uh': [711, 1234, 1235, 1357, 40, 1111, 1234, 1744, 40, 242, 533, 632, 1234, 1744], 'uw': [41, 199, 238, 405, 668, 671, 729, 775, 986, 1020, 1047, 1098, 1222, 1614, 1822, 93, 140, 371, 671, 732, 1221, 1327, 1328, 1384, 1392, 1419, 1491, 1640, 1768, 195, 307, 401, 451, 475, 532, 596, 774, 786, 855, 866, 1030, 1039, 1219, 1295, 1322, 1328, 1361, 1384, 1392, 1589, 1613, 1752, 1753, 1827, 1846, 1862, 1863], 'v': [42, 226, 703, 780, 843, 982, 1266, 1297, 1422, 1738, 1786, 229, 555, 556, 825, 998, 1069, 1246, 1554, 1562, 1759, 1799, 163, 289, 395, 438, 575, 692, 897, 1151, 1160, 1246, 1466, 1522, 1574, 1724], 'vcl': [43, 97, 282, 568, 650, 710, 747, 930, 993, 1064, 1130, 1139, 1355, 1442, 1480, 1591, 1598, 1620, 1624, 1668, 1796, 1818, 150, 201, 301, 349, 409, 459, 548, 569, 629, 706, 890, 1033, 1112, 1128, 1272, 1353, 1481, 1576, 1621, 1623, 1624, 1798, 96, 200, 223, 285, 297, 324, 384, 641, 679, 712, 723, 795, 798, 928, 929, 952, 1134, 1193, 1285, 1588, 1622, 1764], 'w': [44, 133, 476, 543, 637, 714, 725, 1027, 1163, 1265, 1302, 1565, 1611, 1813, 85, 221, 305, 450, 527, 539, 675, 813, 870, 948, 1405, 1526, 1650, 113, 210, 353, 524, 620, 808, 939, 1073, 1166, 1245, 1280, 1435, 1651], 'y': [45, 208, 400, 473, 1592, 1698, 1720, 1780, 1868, 154, 224, 489, 677, 809, 1440, 1778, 1867, 488, 707, 1488, 1741, 1742], 'z': [46, 269, 304, 310, 644, 987, 1050, 1159, 1191, 1259, 1427, 1433, 1482, 1777, 1807, 1816, 73, 128, 176, 439, 500, 776, 881, 904, 1118, 1445, 1672, 1749, 1776, 1811, 65, 156, 311, 390, 419, 428, 923, 953, 988, 1292, 1337, 1379, 1489, 1671, 1705, 1793, 1821], 'zh': [47, 496, 1853, 972, 1853]}





phonemes = ['sil', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'cl', 'd', 'dh', 'dx', 'eh', 'el', 'en', 'epi', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'ix', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'vcl', 'w', 'y', 'z', 'zh']




#Triphones
tritargets = {}
nTriTargets = 0
for n in range(1, 31):
    with open('tri3_ali.'+str(n)+'.pdf.txt') as f:
        triali = [x.strip() for x in f.readlines()]
    for item in triali:
        data = item.split() #Split a string into a list where each word is a list item
        numdata = np.array([int(el) for el in data[1:]]) #here each words become a item
        tritargets[data[0]] = numdata
        nTriTargets = np.max([nTriTargets, numdata.max()])

nTriTargets += 1


trival = {}
nTriVal = 0
for n in range(1, 2):
    with open('tri3_ali_dev.'+str(n)+'.pdf.txt') as f:
        triali = [x.strip() for x in f.readlines()]
    for item in triali:
        data = item.split() #Split a string into a list where each word is a list item
        numdata = np.array([int(el) for el in data[1:]]) #here each words become a item
        trival[data[0]] = numdata
        nTriVal = np.max([nTriVal, numdata.max()])

nTriVal += 1

tritest = {}
nTriTest = 0
for n in range(1, 2):
    with open('tri3_ali_test.'+str(n)+'.pdf.txt') as f:
        triali = [x.strip() for x in f.readlines()]
    for item in triali:
        data = item.split() #Split a string into a list where each word is a list item
        numdata = np.array([int(el) for el in data[1:]]) #here each words become a item
        tritest[data[0]] = numdata
        nTriTest = np.max([nTriTest, numdata.max()])

nTriTest += 1


def frameConcat(x,splice, splType):
    validFrm = int( np.sum(np.sign( np.sum( np.abs(x), axis=1) )) )
    nFrame, nDim = x.shape

    if ( splType == 1):
        spl = splice
        splVec = np.arange(0, int(2*spl+1), 1)
    else:
        spl = int(2*splice)
        splVec = np.arange(0, int(2*spl+1), 2)

    xZerosPad = np.vstack([np.zeros((spl, nDim)), x[0:validFrm,:], np.zeros((spl, nDim))])
    xConc = np.zeros( (validFrm, int(nDim*(2*splice+1))) )

    for iFrm in range(validFrm):
       xConcTmp = np.reshape(xZerosPad[iFrm+splVec,:], (1,int((2*splice+1)*nDim)) )
       xConc[iFrm, :] = xConcTmp
    return xConc

#13 MFCC:
x_tri = np.zeros((0, 143)) #13*11, 5 frames on each side of the current mfcc
x_test = np.zeros((0, 143))
x_val = np.zeros((0, 143))
y_test = np.zeros((0, nTriTest))
y_val = np.zeros((0, nTriVal))



'''
trimodel = Sequential()
trimodel.add(Dense(512, activation='relu', input_shape=(143,)))
trimodel.add(Dense(512, activation='relu'))
trimodel.add(Dropout(0.25))
BatchNormalization(axis=1)
trimodel.add(Dense(512, activation='relu'))
trimodel.add(Dropout(0.25))
trimodel.add(Dense(nTriTargets, activation='softmax'))
trimodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
trimodel.summary()
'''


'''
trimodel = Sequential()
trimodel.add(Dense(1896, activation='relu', input_shape=(143,)))
trimodel.add(Dense(512, activation='relu'))
trimodel.add(Dropout(0.25))
trimodel.add(Dense(512, activation='relu'))
trimodel.add(Dense(nTriTargets, activation='softmax'))
trimodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
trimodel.summary()

'''
trimodel = Sequential()
trimodel.add(Dense(512, activation='relu', input_shape=(143,)))
trimodel.add(Dense(512, activation='relu'))
trimodel.add(Dropout(0.25))
BatchNormalization(axis=1)
trimodel.add(Dense(512, activation='relu'))
trimodel.add(Dropout(0.25))
trimodel.add(Dense(nTriTargets, activation='softmax'))
trimodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
trimodel.summary()


for keys in mfcc_train.keys():
    mfccarray = mfcc_train[keys]
    x_mean = np.mean(mfccarray, axis=0)
    x_std = np.std(mfccarray, axis=0)
    mfcctrain_normalized = ( mfccarray - x_mean ) / x_std
    trainConc=frameConcat(mfcctrain_normalized, 5, 1) #should give 13*11
    x_tri = np.vstack((x_tri, trainConc)) #concatenate mfcc
    
triarray = np.concatenate(list(tritargets.values()))


def tri_generator():
    used_so_far = 0
    batch_size=12929
    #atch_size=203
    OHE = K.one_hot(triarray, nTriTargets)
    while True:
        if (used_so_far < (len(x_tri)-batch_size)+1):
            x_batch = x_tri[used_so_far:(used_so_far + batch_size), :]
            y_batch = OHE[used_so_far:(used_so_far + batch_size), :]
            yield(x_batch, y_batch)
            used_so_far += batch_size
        else:
            used_so_far = 0



for keys in mfcc_val.keys():
        valarray = mfcc_val[keys]
        mfccval_normalized = (valarray - x_mean ) / x_std
        valConc=frameConcat(mfccval_normalized, 5, 1)
        x_val = np.vstack((x_val, valConc))

        target_trival = trival[keys]
        Labels_val = np.eye(nTriVal)
        val_OHE = Labels_val[target_trival, :]
        y_val = np.vstack((y_val, val_OHE))


def validation_generator():
    used_so_far = 0
    batch_size=12929
    while True:
        if (used_so_far < (len(x_val)-batch_size)+1):
            x_batch = x_val[used_so_far:(used_so_far + batch_size), :]
            y_batch = y_val[used_so_far:(used_so_far + batch_size)]
            yield(x_batch, y_batch)
            used_so_far += batch_size
        else:
            used_so_far = 0



for keys in mfcc_test.keys():
    testarray = mfcc_test[keys]
    mfcctest_normalized = ( testarray - x_mean ) / x_std
    testConc=frameConcat(mfcctest_normalized, 5, 1)
    x_test = np.vstack((x_test, testConc))

    tri_test = tritest[keys]
    Labels_test = np.eye(nTriTest)
    test_OHE = Labels_test[tri_test, :]
    y_test = np.vstack((y_test, test_OHE))




#print(y_val.shape) (122487, 1896)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
#history = trimodel.fit_generator(generator=(tri_generator()), steps_per_epoch=np.ceil(len(x_tri)/12929), epochs=20, callbacks=[callback], use_multiprocessing=False, validation_data=validation_generator(), validation_steps=np.ceil(len(x_val)/12929), shuffle=True)
history = trimodel.fit_generator(generator=(tri_generator()), steps_per_epoch=np.ceil(len(x_tri)/12929), epochs=20, callbacks=[callback], use_multiprocessing=False, validation_data=(x_val, y_val), shuffle=True)



fig, ax = plt.subplots()
plt.plot(history.history['accuracy'], marker="*", label='Train accuracy', color='blue')
plt.plot(history.history['val_accuracy'], marker="o", label='Test accuracy', color='red')
plt.title('Frame accuracy rate' )
plt.ylabel('Accuracy rate (%)')
plt.xlabel('Epoch')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=None,symbol=None))
plt.legend(loc='best')
plt.show()
plt.savefig('TRISS20.png')


#Test model: (on full data set)
score, acc = trimodel.evaluate(x_test, y_test, verbose=0)
print('Test score:', score)
print('Test accuracy:', acc)


#predicted =  trimodel.predict(x_test)

targetClass =  np.concatenate(list(tritest.values()))
predictedClass = trimodel.predict_classes(x_test)

print('len target:', len(targetClass))
print('len predict:', len(predictedClass))


#MAKES LIST OF LIST FROM DICTIONARY
statesValues = []
for keys in states.keys():
    phonemeStates = states[keys] #this is a list
    statesValues.append(phonemeStates)

print('len statesValues', len(statesValues))

#############################   PHONEME RECOGNITION TEST SET  #########################
stateInPhoneme = []
correctIndexStates = []
for i in range(len(x_test)):
    if (predictedClass[i] != targetClass[i]):
        for index, nested_list in enumerate(statesValues):
            if predictedClass[i] in nested_list and targetClass[i] in nested_list:
                stateInPhoneme.append(i)

###Recalculating accuracy
correct = 0
for j in range(len(x_test)):
    if predictedClass[j] == targetClass[j]:
        correct +=1
print('correct: ', correct)

#print(stateInPhoneme)
print('Amount of states in phoneme when the predicted and target is not the same: ', len(stateInPhoneme))

correctPhonemes = len(stateInPhoneme)+correct
print(correctPhonemes)

#Phoneme recognition:
newAccuracy = 100 * (correctPhonemes/len(predictedClass))
print('Phoneme recognition accuracy TEST SET: ', newAccuracy)

###################   END PHONEME RECOGNITION   #####################



'''
#############################   PHONEME RECOGNITION TRAIN SET  #########################
predictedClass2 = trimodel.predict_classes(x_tri)


stateInPhoneme = []
correctIndexStates = []
for i in range(len(x_tri)):
    if (predictedClass2[i] != triarray[i]):
        for index, nested_list in enumerate(statesValues):
            if predictedClass2[i] in nested_list and triarray[i] in nested_list:
                stateInPhoneme.append(i)

###Recalculating accuracy
correct = 0
for j in range(len(x_tri)):
    if predictedClass2[j] == triarray[j]:
        correct +=1
print('correct: ', correct)

#print(stateInPhoneme)
print('Amount of states in phoneme when the predicted and target is not the same: ', len(stateInPhoneme))

correctPhonemes = len(stateInPhoneme)+correct
print(correctPhonemes)

#Phoneme recognition:
newAccuracy = 100 * (correctPhonemes/len(predictedClass2))
print('Phoneme recognition accuracy TRAIN SET: ', newAccuracy)

###################   END PHONEME RECOGNITION   #####################
'''




###################   SEMI-SUPERVISED LEARNING  #####################
predTarget =trimodel.predict(x_tri)

trimodel.save('modelTrimono.h5')
del trimodel

student2 = load_model('modelTrimono.h5')
student2.summary()

def student_generator():
    used_so_far = 0
    batch_size=12929
    while True:
        if (used_so_far < (len(x_val)-batch_size)+1):
            x_batch = x_tri[used_so_far:(used_so_far + batch_size), :]
            y_batch = predTarget[used_so_far:(used_so_far + batch_size)]
            yield(x_batch, y_batch)
            used_so_far += batch_size
        else:
            used_so_far = 0


callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
history2 = student2.fit_generator(generator=(student_generator()), steps_per_epoch=np.ceil(len(x_tri)/12929), epochs=20, use_multiprocessing=False,  validation_data=(x_val, y_val), callbacks=[callback] , shuffle=True)


fig, ax = plt.subplots()
plt.plot(history2.history['accuracy'], marker="*", label='Train accuracy', color='blue')
plt.plot(history2.history['val_accuracy'], marker="o", label='Test accuracy', color='red')
plt.title('Frame accuracy rate' )
plt.ylabel('Accuracy rate (%)')
plt.xlabel('Epoch')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=None, symbol=None))
#ax.grid()
plt.legend(loc='best')
plt.savefig('TRISSL20.png')


score, acc = student2.evaluate(x_test, y_test)
print('Test score: ', score)
print('Test accuracy: ', acc)



#########################   #PHONEME RECOGNITION SSL   #########################

predictedClass = student2.predict_classes(x_test)
stateInPhoneme = []
for i in range(len(y_test)):
    if (predictedClass[i] != targetClass[i]):
        for index, nested_list in enumerate(statesValues):
            if predictedClass[i] in nested_list and targetClass[i] in nested_list:
                stateInPhoneme.append(i)
                #print(index)

###Recalculating accuracy
correct = 0
for j in range(len(y_test)):
    #for j in range(len(x_train)):
    if predictedClass[j] == targetClass[j]:
        correct +=1
print('correct: ', correct)

#print(stateInPhoneme)
print('Amount of states in phoneme when the predicted and target is not the same: ', len(stateInPhoneme))

correctPhonemes = len(stateInPhoneme)+correct
print(correctPhonemes)

#Phoneme recognition:
newAccuracy = 100 * (correctPhonemes/len(predictedClass))
print('Phoneme recognition accuracy: ', newAccuracy)
###################### PHONEME RECOGNITION #####################

print('len target ssl:', len(targetClass))
print('len predict ssl:', len(predictedClass))

posPhoneme = []
posTarget = []
for i in range(len(predictedClass)):
    for index, nested_list in enumerate(statesValues):
        if predictedClass[i] in nested_list:
            posPhoneme.append(index)

        if targetClass[i] in nested_list:
            posTarget.append(index)

print('posPhoneme: ', len(posPhoneme))
print('posTarget: ', len(posTarget))

y_pred = posPhoneme[0:len(posTarget)]
y = posTarget

print('lenght of predicted cm:', len(y_pred))
print('length target cm: ', len(y))

#Correctly predicted color CM
cm=confusion_matrix(y, y_pred)

#print(cm)
cm2 = cm+10**(-10)

'''
plt.figure(figsize=(31, 22))
#plt.figure(figsize=(27, 27))
plt.imshow(cm2, norm=colors.LogNorm(vmin=cm2.min(), vmax=cm2.max()))
tick_marks = np.arange(len(phonemes))
plt.xticks(tick_marks, phonemes, fontsize=22)
plt.yticks(tick_marks, phonemes, fontsize=22)
plt.tight_layout()
plt.savefig('cmtrimfccgiampi2.png')


cm_df = pd.DataFrame(cm, index = phonemes, columns = phonemes)
cm_df2 = np.log(cm_df)
plt.figure(figsize=(31, 22))
plt.imshow(cm_df2, interpolation='nearest', aspect='auto', cmap='Reds')
tick_marks = np.arange(len(phonemes))
plt.xticks(tick_marks, phonemes, fontsize=17)
plt.yticks(tick_marks, phonemes, fontsize=17)
plt.colorbar()
plt.savefig('CMtriSSL.png')
'''




cm_df = pd.DataFrame(cm, index = phonemes, columns = phonemes)
cm_df2 = np.log(cm_df)
plt.figure(figsize=(10, 7))
plt.imshow(cm_df2, interpolation='nearest', aspect='auto', cmap='Greens')
tick_marks = np.arange(len(phonemes))
plt.xticks(tick_marks, phonemes, fontsize=8)
plt.yticks(tick_marks, phonemes, fontsize=8)
plt.savefig('CMtri.png')




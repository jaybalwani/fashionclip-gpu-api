import mysql.connector

# def connection(): 

#     conn = None
#     try:
#         conn = mysql.connector.connect(
#                 host="sai-db.c5os8oqs26uu.ap-south-1.rds.amazonaws.com",
#                 user="admin",
#                 password="891OtAbLcKAm6pvKDfRs",
#                 database="sai",
#         )

#         if conn is not None and conn.is_connected():
#             print("Database connection established successfully")
#             cursor = conn.cursor()
#             return conn, cursor
#         else:
#             print("Failed to establish a database connection")
#             return None, None

#     except mysql.connector.Error as e:
#         print(f"Error while connecting to MySQL: {e}")
#         return None, None


def connection(**kwargs): 

    conn = None
    try:
        conn = mysql.connector.connect(
                host="sai-db.c5os8oqs26uu.ap-south-1.rds.amazonaws.com",
                user="admin",
                # password="891OtAbLcKAm6pvKDfRs",
                password="4dGDVgNKWJG2OfxPyJtH",
                database="sai",
        )

        if conn is not None and conn.is_connected():
            print("Database connection established successfully")
            if kwargs and kwargs.get('dictFlag', {}) == True:
                cursor = conn.cursor(dictionary=True)
            else:
                cursor = conn.cursor()
            return conn, cursor
            # cursor = conn.cursor()
            # return conn, cursor
        else:
            print("Failed to establish a database connection")
            return None, None

    except mysql.connector.Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return None, None

import sqlite3
import datetime
from os import path

# DATABASE
class db:

    # TABLE
    class Table:

        def __init__(self, dataBase, dbSuper, name, create=False, columns=None):
            
            # database connection
            self.dataBase = dataBase
            cur = dataBase.cursor()

            # database class
            self.dbSuper = dbSuper
            
            # if new table being created
            if create:
                cmd = f"CREATE TABLE {name}("
                
                # add column names
                for i in columns:
                    cmd += str(i) + ", "
                cmd = cmd[:-2] + ")"

                # print commands if applicable
                if self.dbSuper.printCmds:
                    time = datetime.datetime.strftime(datetime.datetime.now(), "%m/%d/%y %H:%M:%S")
                    print(f"{time} | {cmd}")
                
                # execute and update
                cur.execute(cmd)
                dataBase.commit()
                self.columns = columns
                self.name = name
        
            else:
                # if this table is based on a prexisting on in database
                self.name = name[0]
                self.columns = [i[1] for i in cur.execute(f"PRAGMA table_info({name[0]})").fetchall()]
                self.dataBase.commit()
                
        def insert(self, vals):

            # add a row
            cur = self.dataBase.cursor()
            cmd = f"INSERT INTO {self.name} VALUES ("

            # check for string and null datatypes (strings need " ")
            for i in vals:
                if type(i) == type("string") and i != "NULL":
                    i = f"'{i}'"
                if i is not None:
                    cmd += str(i) + ", "
                else:
                    cmd += "NULL, "
            cmd = cmd[:-2] + f")"

            # print if applicable
            if self.dbSuper.printCmds:
                time = datetime.datetime.strftime(datetime.datetime.now(), "%m/%d/%y %H:%M:%S")
                print(f"{time} | {cmd}")

            # execute
            cur.execute(cmd)
            self.dataBase.commit()
        
        """
        def insertRows(self, vals):
            
            # insert multiple rows
            cur = self.dataBase.cursor()
            for val in vals:

                # take care of strings and nulls
                cmd = f"INSERT INTO {self.name} VALUES ("
                for i in val:
                    if type(i) == type("string") and i != "NULL":
                        i = f"'{i}'"
                    if i is not None:
                        cmd += str(i) + ", "
                    else:
                        cmd += "NULL, "
                cmd = cmd[:-2] + ")"

                # print if applicable
                if self.dbSuper.printCmds:
                    time = datetime.datetime.strftime(datetime.datetime.now(), "%m/%d/%y %H:%M:%S")
                    print(f"{time} | {cmd}")

                # execute
                cur.execute(cmd)
                self.dataBase.commit()
        """

        def insertRows(self, vals):
            
            # insert multiple rows
            cur = self.dataBase.cursor()

            cmd = f"INSERT INTO {self.name} VALUES "

            for val in vals:
                
                cmd += "("

                # take care of strings and nulls
                for i in val:
                    if type(i) == type("string") and i != "NULL":
                        i = f"'{i}'"
                    if i is not None:
                        cmd += str(i) + ", "
                    else:
                        cmd += "NULL, "
                cmd = cmd[:-2] + "), "

            # execute
            cmd = cmd[:-2]
            cmd += ";"
            try:
                cur.execute(cmd)
                self.dataBase.commit()
            except sqlite3.OperationalError:
                print(cmd)
                return -1

            # print if applicable
            if self.dbSuper.printCmds:
                time = datetime.datetime.strftime(datetime.datetime.now(), "%m/%d/%y %H:%M:%S")
                print(f"{time} | {cmd}")
            
            return 0
            
        
        def getAll(self):
            
            # get all data from table
            cur = self.dataBase.cursor()
            rows = cur.execute(f"SELECT * FROM {self.name}").fetchall()
            self.dataBase.commit()
            return rows
        
        def get(self, column):

            # get a column of data from a table
            cur = self.dataBase.cursor()
            rows = cur.execute(f"SELECT {column} FROM {self.name}").fetchall()
            self.dataBase.commit()
            return rows

        def getAllWhere(self, criteriaColumn, criteria):

            # get all where criteria is met
            cur = self.dataBase.cursor()
            if type(criteria) == type("string"):
                criteria = f"'{criteria}'"
            rows = cur.execute(f"SELECT * FROM {self.name} WHERE {criteriaColumn} = {criteria}").fetchall()
            self.dataBase.commit()
            return rows
        
        def deleteWhere(self, column, columnValue):

            # delete all where criteria is met
            cur = self.dataBase.cursor()
            if type(columnValue) == type("string"):
                    columnValue = f"'{columnValue}'"
            cmd = f'DELETE FROM {self.name} WHERE {column}={columnValue}'

            # print if applicable
            if self.dbSuper.printCmds:
                time = datetime.datetime.strftime(datetime.datetime.now(), "%m/%d/%y %H:%M:%S")
                print(f"{time} | {cmd}")

            # execute
            cur.execute(cmd).fetchall()
            self.dataBase.commit()
        
        def empty(self):

            # delete all values in table
            cur = self.dataBase.cursor()
            cmd = f'DELETE FROM {self.name}'

            # print if applicable
            if self.dbSuper.printCmds:
                time = datetime.datetime.strftime(datetime.datetime.now(), "%m/%d/%y %H:%M:%S")
                print(f"{time} | {cmd}")

            # execute
            cur.execute(cmd).fetchall()
            self.dataBase.commit()
        
        def destroy(self):

            # remove table from db
            cur = self.dataBase.cursor()
            cmd = f'DROP TABLE {self.name}'

            # print if applicable
            if self.dbSuper.printCmds:
                time = datetime.datetime.strftime(datetime.datetime.now(), "%m/%d/%y %H:%M:%S")
                print(f"{time} | {cmd}")

            # execute
            cur.execute(cmd).fetchall()
            self.dataBase.commit()

            # remove self from db.table
            self.dbSuper.table.pop(f'{self.name}')
        
        def writeFirst(self, cols):

            # clear table and replace the first line
            # for tracking data that only requires one row
            self.empty()
            self.insert(cols)

    def __init__(self, name, dir = None):

        # info
        self.name = name
        self.printCmds = True
        if dir is None:
            self.con = sqlite3.connect(self.name)
        else:
            self.con = sqlite3.connect(path.join(dir, self.name))
        cur = self.con.cursor()

        # {table_name : table object, }
        self.table: dict[str: self.Table] = {}

        # make a table for each tablename
        tableNames = cur.execute("SELECT name FROM sqlite_schema WHERE type='table'").fetchall()
        for i in tableNames:
            self.table[i[0]] = self.Table(self.con, self, i)
        self.con.commit()

    def newTable(self, tableName, columns):

        # table object
        table = self.Table(self.con, self, tableName, True, columns)

        # add to self.table
        self.table[tableName] = table

        return table

    def sql(self, command):
        cur = self.con.cursor()
        ret = cur.execute(command).fetchall()
        self.con.commit()
        return ret

        

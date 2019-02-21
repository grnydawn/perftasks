from fparser.two.Fortran2003 import (Subroutine_Subprogram, Function_Subprogram,
        Main_Program, Main_Program0)

upper_stmts = (Subroutine_Subprogram, Function_Subprogram,
        Main_Program, Main_Program0)

def find_upper_stmt(stmt):

    parent = getattr(stmt, "parent", None)

    while parent is not None:
        if isinstance(parent, upper_stmts):
            return parent
        parent = getattr(parent, "parent", None)

    raise Exception("Could not find an upper statement.")

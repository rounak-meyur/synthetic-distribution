# -*- coding: utf-8 -*-
"""
 Copyright 2019, Gurobi Optimization, LLC

   This example reads a model from a file, sets up a callback that
   monitors optimization progress and implements a custom
   termination strategy, and outputs progress information to the
   screen and to a log file.

   The termination strategy implemented in this callback stops the
   optimization of a MIP model once at least one of the following two
   conditions have been satisfied:
     1) The optimality gap is less than 10%
     2) At least 10000 nodes have been explored, and an integer feasible
        solution has been found.
   Note that termination is normally handled through Gurobi parameters
   (MIPGap, NodeLimit, etc.).  You should only use a callback for
   termination if the available parameters don't capture your desired
   termination criterion.
"""


import gurobipy as grb

# Define my callback function

def mycallback(model, where):
    if where == grb.GRB.Callback.POLLING:
        # Ignore polling callback
        pass
    elif where == grb.GRB.Callback.PRESOLVE:
        # Presolve callback
        cdels = model.cbGet(grb.GRB.Callback.PRE_COLDEL)
        rdels = model.cbGet(grb.GRB.Callback.PRE_ROWDEL)
        if cdels or rdels:
            print('%d columns and %d rows are removed' % (cdels, rdels))
    elif where == grb.GRB.Callback.SIMPLEX:
        # Simplex callback
        itcnt = model.cbGet(grb.GRB.Callback.SPX_ITRCNT)
        if itcnt - model._lastiter >= 100:
            model._lastiter = itcnt
            obj = model.cbGet(grb.GRB.Callback.SPX_OBJVAL)
            ispert = model.cbGet(grb.GRB.Callback.SPX_ISPERT)
            pinf = model.cbGet(grb.GRB.Callback.SPX_PRIMINF)
            dinf = model.cbGet(grb.GRB.Callback.SPX_DUALINF)
            if ispert == 0:
                ch = ' '
            elif ispert == 1:
                ch = 'S'
            else:
                ch = 'P'
            print('%d %g%s %g %g' % (int(itcnt), obj, ch, pinf, dinf))
    elif where == grb.GRB.Callback.MIP:
        # General MIP callback
        nodecnt = model.cbGet(grb.GRB.Callback.MIP_NODCNT)
        objbst = model.cbGet(grb.GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(grb.GRB.Callback.MIP_OBJBND)
        solcnt = model.cbGet(grb.GRB.Callback.MIP_SOLCNT)
        if nodecnt - model._lastnode >= 100:
            model._lastnode = nodecnt
            actnodes = model.cbGet(grb.GRB.Callback.MIP_NODLFT)
            itcnt = model.cbGet(grb.GRB.Callback.MIP_ITRCNT)
            cutcnt = model.cbGet(grb.GRB.Callback.MIP_CUTCNT)
            print('%d %d %d %g %g %d %d' % (nodecnt, actnodes, \
                  itcnt, objbst, objbnd, solcnt, cutcnt))
        if abs(objbst - objbnd) < 0.1 * (1.0 + abs(objbst)):
            print('Stop early - 10% gap achieved')
            model.terminate()
        if nodecnt >= 10000 and solcnt:
            print('Stop early - 10000 nodes explored')
            model.terminate()
    elif where == grb.GRB.Callback.MIPSOL:
        # MIP solution callback
        nodecnt = model.cbGet(grb.GRB.Callback.MIPSOL_NODCNT)
        obj = model.cbGet(grb.GRB.Callback.MIPSOL_OBJ)
        solcnt = model.cbGet(grb.GRB.Callback.MIPSOL_SOLCNT)
        x = model.cbGetSolution(model._vars)
        print('**** New solution at node %d, obj %g, sol %d, ' \
              'x[0] = %g ****' % (nodecnt, obj, solcnt, x[0]))
    elif where == grb.GRB.Callback.MIPNODE:
        # MIP node callback
        print('**** New node ****')
        if model.cbGet(grb.GRB.Callback.MIPNODE_STATUS) == grb.GRB.Status.OPTIMAL:
            x = model.cbGetNodeRel(model._vars)
            model.cbSetSolution(model.getVars(), x)
    elif where == grb.GRB.Callback.BARRIER:
        # Barrier callback
        itcnt = model.cbGet(grb.GRB.Callback.BARRIER_ITRCNT)
        primobj = model.cbGet(grb.GRB.Callback.BARRIER_PRIMOBJ)
        dualobj = model.cbGet(grb.GRB.Callback.BARRIER_DUALOBJ)
        priminf = model.cbGet(grb.GRB.Callback.BARRIER_PRIMINF)
        dualinf = model.cbGet(grb.GRB.Callback.BARRIER_DUALINF)
        cmpl = model.cbGet(grb.GRB.Callback.BARRIER_COMPL)
        print('%d %g %g %g %g %g' % (itcnt, primobj, dualobj, \
              priminf, dualinf, cmpl))
    elif where == grb.GRB.Callback.MESSAGE:
        # Message callback
        msg = model.cbGet(grb.GRB.Callback.MSG_STRING)
        model._logfile.write(msg)




./abstract/finite.vo ./abstract/finite.glob ./abstract/finite.v.beautified: ./abstract/finite.v ./abstract/bibli.vo
./abstract/finite.vio: ./abstract/finite.v ./abstract/bibli.vio
./abstract/abstract_machine.vo ./abstract/abstract_machine.glob ./abstract/abstract_machine.v.beautified: ./abstract/abstract_machine.v ./abstract/fifo.vo ./abstract/table.vo ./abstract/reduce.vo ./abstract/sigma2.vo
./abstract/abstract_machine.vio: ./abstract/abstract_machine.v ./abstract/fifo.vio ./abstract/table.vio ./abstract/reduce.vio ./abstract/sigma2.vio
./abstract/sigma2.vo ./abstract/sigma2.glob ./abstract/sigma2.v.beautified: ./abstract/sigma2.v ./abstract/fifo.vo ./abstract/table.vo ./abstract/reduce.vo
./abstract/sigma2.vio: ./abstract/sigma2.v ./abstract/fifo.vio ./abstract/table.vio ./abstract/reduce.vio
./abstract/table2.vo ./abstract/table2.glob ./abstract/table2.v.beautified: ./abstract/table2.v ./abstract/table.vo
./abstract/table2.vio: ./abstract/table2.v ./abstract/table.vio
./abstract/sum.vo ./abstract/sum.glob ./abstract/sum.v.beautified: ./abstract/sum.v ./abstract/reduce.vo ./abstract/table.vo ./abstract/abstract_machine.vo
./abstract/sum.vio: ./abstract/sum.v ./abstract/reduce.vio ./abstract/table.vio ./abstract/abstract_machine.vio
./abstract/table.vo ./abstract/table.glob ./abstract/table.v.beautified: ./abstract/table.v ./abstract/finite.vo
./abstract/table.vio: ./abstract/table.v ./abstract/finite.vio
./abstract/reduce.vo ./abstract/reduce.glob ./abstract/reduce.v.beautified: ./abstract/reduce.v ./abstract/fifo.vo
./abstract/reduce.vio: ./abstract/reduce.v ./abstract/fifo.vio
./abstract/fifo.vo ./abstract/fifo.glob ./abstract/fifo.v.beautified: ./abstract/fifo.v ./abstract/bibli.vo
./abstract/fifo.vio: ./abstract/fifo.v ./abstract/bibli.vio
./abstract/bibli.vo ./abstract/bibli.glob ./abstract/bibli.v.beautified: ./abstract/bibli.v
./abstract/bibli.vio: ./abstract/bibli.v
expose.vo expose.glob expose.v.beautified: expose.v
expose.vio: expose.v
./machine2/invariant4.vo ./machine2/invariant4.glob ./machine2/invariant4.v.beautified: ./machine2/invariant4.v ./machine2/invariant0.vo ./machine2/invariant1.vo ./machine2/invariant2.vo ./machine2/invariant3.vo
./machine2/invariant4.vio: ./machine2/invariant4.v ./machine2/invariant0.vio ./machine2/invariant1.vio ./machine2/invariant2.vio ./machine2/invariant3.vio
./machine2/invariant2.vo ./machine2/invariant2.glob ./machine2/invariant2.v.beautified: ./machine2/invariant2.v ./machine2/machine.vo ./machine2/cardinal.vo ./machine2/comm.vo
./machine2/invariant2.vio: ./machine2/invariant2.v ./machine2/machine.vio ./machine2/cardinal.vio ./machine2/comm.vio
./machine2/liveness.vo ./machine2/liveness.glob ./machine2/liveness.v.beautified: ./machine2/liveness.v ./machine2/invariant8.vo
./machine2/liveness.vio: ./machine2/liveness.v ./machine2/invariant8.vio
./machine2/invariant8.vo ./machine2/invariant8.glob ./machine2/invariant8.v.beautified: ./machine2/invariant8.v ./machine2/invariant5.vo ./machine2/invariant7.vo
./machine2/invariant8.vio: ./machine2/invariant8.v ./machine2/invariant5.vio ./machine2/invariant7.vio
./machine2/cardinal.vo ./machine2/cardinal.glob ./machine2/cardinal.v.beautified: ./machine2/cardinal.v ./abstract/reduce.vo ./machine2/machine.vo
./machine2/cardinal.vio: ./machine2/cardinal.v ./abstract/reduce.vio ./machine2/machine.vio
./machine2/invariant6.vo ./machine2/invariant6.glob ./machine2/invariant6.v.beautified: ./machine2/invariant6.v ./machine2/invariant0.vo ./machine2/invariant1.vo ./machine2/invariant2.vo ./machine2/invariant3.vo ./machine2/invariant4.vo
./machine2/invariant6.vio: ./machine2/invariant6.v ./machine2/invariant0.vio ./machine2/invariant1.vio ./machine2/invariant2.vio ./machine2/invariant3.vio ./machine2/invariant4.vio
./machine2/invariant3.vo ./machine2/invariant3.glob ./machine2/invariant3.v.beautified: ./machine2/invariant3.v ./machine2/invariant0.vo ./machine2/invariant1.vo ./machine2/invariant2.vo
./machine2/invariant3.vio: ./machine2/invariant3.v ./machine2/invariant0.vio ./machine2/invariant1.vio ./machine2/invariant2.vio
./machine2/alternate.vo ./machine2/alternate.glob ./machine2/alternate.v.beautified: ./machine2/alternate.v ./machine2/machine.vo ./machine2/cardinal.vo ./machine2/comm.vo
./machine2/alternate.vio: ./machine2/alternate.v ./machine2/machine.vio ./machine2/cardinal.vio ./machine2/comm.vio
./machine2/comm.vo ./machine2/comm.glob ./machine2/comm.v.beautified: ./machine2/comm.v ./abstract/reduce.vo ./machine2/cardinal.vo ./abstract/sigma2.vo ./abstract/sum.vo
./machine2/comm.vio: ./machine2/comm.v ./abstract/reduce.vio ./machine2/cardinal.vio ./abstract/sigma2.vio ./abstract/sum.vio
./machine2/invariant7.vo ./machine2/invariant7.glob ./machine2/invariant7.v.beautified: ./machine2/invariant7.v ./machine2/invariant6.vo
./machine2/invariant7.vio: ./machine2/invariant7.v ./machine2/invariant6.vio
./machine2/invariant0.vo ./machine2/invariant0.glob ./machine2/invariant0.v.beautified: ./machine2/invariant0.v ./machine2/machine.vo ./machine2/cardinal.vo ./machine2/comm.vo
./machine2/invariant0.vio: ./machine2/invariant0.v ./machine2/machine.vio ./machine2/cardinal.vio ./machine2/comm.vio
./machine2/invariant1.vo ./machine2/invariant1.glob ./machine2/invariant1.v.beautified: ./machine2/invariant1.v ./machine2/machine.vo ./machine2/cardinal.vo ./machine2/comm.vo
./machine2/invariant1.vio: ./machine2/invariant1.v ./machine2/machine.vio ./machine2/cardinal.vio ./machine2/comm.vio
./machine2/machine.vo ./machine2/machine.glob ./machine2/machine.v.beautified: ./machine2/machine.v ./abstract/abstract_machine.vo
./machine2/machine.vio: ./machine2/machine.v ./abstract/abstract_machine.vio
./machine2/invariant5.vo ./machine2/invariant5.glob ./machine2/invariant5.v.beautified: ./machine2/invariant5.v ./machine2/alternate.vo ./machine2/invariant0.vo ./machine2/invariant1.vo ./machine2/invariant2.vo ./machine2/invariant3.vo ./machine2/invariant4.vo
./machine2/invariant5.vio: ./machine2/invariant5.v ./machine2/alternate.vio ./machine2/invariant0.vio ./machine2/invariant1.vio ./machine2/invariant2.vio ./machine2/invariant3.vio ./machine2/invariant4.vio
./machine3/invariant4.vo ./machine3/invariant4.glob ./machine3/invariant4.v.beautified: ./machine3/invariant4.v ./machine3/invariant0.vo ./machine3/invariant1.vo ./machine3/invariant2.vo ./machine3/invariant3.vo
./machine3/invariant4.vio: ./machine3/invariant4.v ./machine3/invariant0.vio ./machine3/invariant1.vio ./machine3/invariant2.vio ./machine3/invariant3.vio
./machine3/invariant2.vo ./machine3/invariant2.glob ./machine3/invariant2.v.beautified: ./machine3/invariant2.v ./machine3/machine.vo ./machine3/cardinal.vo ./machine3/comm.vo ./machine3/still_to_prove.vo
./machine3/invariant2.vio: ./machine3/invariant2.v ./machine3/machine.vio ./machine3/cardinal.vio ./machine3/comm.vio ./machine3/still_to_prove.vio
./machine3/still_to_prove.vo ./machine3/still_to_prove.glob ./machine3/still_to_prove.v.beautified: ./machine3/still_to_prove.v ./machine3/machine.vo ./machine3/cardinal.vo ./machine3/comm.vo
./machine3/still_to_prove.vio: ./machine3/still_to_prove.v ./machine3/machine.vio ./machine3/cardinal.vio ./machine3/comm.vio
./machine3/cardinal.vo ./machine3/cardinal.glob ./machine3/cardinal.v.beautified: ./machine3/cardinal.v ./abstract/reduce.vo ./machine3/machine.vo
./machine3/cardinal.vio: ./machine3/cardinal.v ./abstract/reduce.vio ./machine3/machine.vio
./machine3/invariant6.vo ./machine3/invariant6.glob ./machine3/invariant6.v.beautified: ./machine3/invariant6.v ./machine3/invariant0.vo ./machine3/invariant1.vo ./machine3/invariant2.vo ./machine3/invariant3.vo ./machine3/invariant4.vo
./machine3/invariant6.vio: ./machine3/invariant6.v ./machine3/invariant0.vio ./machine3/invariant1.vio ./machine3/invariant2.vio ./machine3/invariant3.vio ./machine3/invariant4.vio
./machine3/invariant3.vo ./machine3/invariant3.glob ./machine3/invariant3.v.beautified: ./machine3/invariant3.v ./machine3/invariant0.vo ./machine3/invariant1.vo ./machine3/invariant2.vo
./machine3/invariant3.vio: ./machine3/invariant3.v ./machine3/invariant0.vio ./machine3/invariant1.vio ./machine3/invariant2.vio
./machine3/alternate.vo ./machine3/alternate.glob ./machine3/alternate.v.beautified: ./machine3/alternate.v ./machine3/machine.vo ./machine3/cardinal.vo ./machine3/comm.vo
./machine3/alternate.vio: ./machine3/alternate.v ./machine3/machine.vio ./machine3/cardinal.vio ./machine3/comm.vio
./machine3/comm.vo ./machine3/comm.glob ./machine3/comm.v.beautified: ./machine3/comm.v ./abstract/reduce.vo ./machine3/cardinal.vo ./abstract/sigma2.vo ./abstract/sum.vo
./machine3/comm.vio: ./machine3/comm.v ./abstract/reduce.vio ./machine3/cardinal.vio ./abstract/sigma2.vio ./abstract/sum.vio
./machine3/invariant0.vo ./machine3/invariant0.glob ./machine3/invariant0.v.beautified: ./machine3/invariant0.v ./machine3/machine.vo ./machine3/cardinal.vo ./machine3/comm.vo ./machine3/still_to_prove.vo
./machine3/invariant0.vio: ./machine3/invariant0.v ./machine3/machine.vio ./machine3/cardinal.vio ./machine3/comm.vio ./machine3/still_to_prove.vio
./machine3/invariant1.vo ./machine3/invariant1.glob ./machine3/invariant1.v.beautified: ./machine3/invariant1.v ./machine3/machine.vo ./machine3/cardinal.vo ./machine3/comm.vo ./machine3/still_to_prove.vo
./machine3/invariant1.vio: ./machine3/invariant1.v ./machine3/machine.vio ./machine3/cardinal.vio ./machine3/comm.vio ./machine3/still_to_prove.vio
./machine3/machine.vo ./machine3/machine.glob ./machine3/machine.v.beautified: ./machine3/machine.v ./abstract/abstract_machine.vo
./machine3/machine.vio: ./machine3/machine.v ./abstract/abstract_machine.vio
./machine3/invariant5.vo ./machine3/invariant5.glob ./machine3/invariant5.v.beautified: ./machine3/invariant5.v ./machine3/alternate.vo ./machine3/invariant0.vo ./machine3/invariant1.vo ./machine3/invariant2.vo ./machine3/invariant3.vo ./machine3/invariant4.vo
./machine3/invariant5.vio: ./machine3/invariant5.v ./machine3/alternate.vio ./machine3/invariant0.vio ./machine3/invariant1.vio ./machine3/invariant2.vio ./machine3/invariant3.vio ./machine3/invariant4.vio
./machine0/table_act.vo ./machine0/table_act.glob ./machine0/table_act.v.beautified: ./machine0/table_act.v ./machine0/machine.vo
./machine0/table_act.vio: ./machine0/table_act.v ./machine0/machine.vio
./machine0/init.vo ./machine0/init.glob ./machine0/init.v.beautified: ./machine0/init.v ./machine0/machine.vo ./machine0/counting.vo
./machine0/init.vio: ./machine0/init.v ./machine0/machine.vio ./machine0/counting.vio
./machine0/rece_copy3.vo ./machine0/rece_copy3.glob ./machine0/rece_copy3.v.beautified: ./machine0/rece_copy3.v ./machine0/init.vo ./machine0/table_act.vo ./machine0/mess_act.vo
./machine0/rece_copy3.vio: ./machine0/rece_copy3.v ./machine0/init.vio ./machine0/table_act.vio ./machine0/mess_act.vio
./machine0/rece_copy1.vo ./machine0/rece_copy1.glob ./machine0/rece_copy1.v.beautified: ./machine0/rece_copy1.v ./machine0/init.vo ./machine0/table_act.vo ./machine0/mess_act.vo
./machine0/rece_copy1.vio: ./machine0/rece_copy1.v ./machine0/init.vio ./machine0/table_act.vio ./machine0/mess_act.vio
./machine0/counting.vo ./machine0/counting.glob ./machine0/counting.v.beautified: ./machine0/counting.v ./machine0/machine.vo
./machine0/counting.vio: ./machine0/counting.v ./machine0/machine.vio
./machine0/rece_inc.vo ./machine0/rece_inc.glob ./machine0/rece_inc.v.beautified: ./machine0/rece_inc.v ./machine0/init.vo ./machine0/table_act.vo ./machine0/mess_act.vo
./machine0/rece_inc.vio: ./machine0/rece_inc.v ./machine0/init.vio ./machine0/table_act.vio ./machine0/mess_act.vio
./machine0/rece_dec.vo ./machine0/rece_dec.glob ./machine0/rece_dec.v.beautified: ./machine0/rece_dec.v ./machine0/init.vo ./machine0/table_act.vo ./machine0/mess_act.vo
./machine0/rece_dec.vio: ./machine0/rece_dec.v ./machine0/init.vio ./machine0/table_act.vio ./machine0/mess_act.vio
./machine0/copy.vo ./machine0/copy.glob ./machine0/copy.v.beautified: ./machine0/copy.v ./machine0/init.vo ./machine0/table_act.vo ./machine0/mess_act.vo
./machine0/copy.vio: ./machine0/copy.v ./machine0/init.vio ./machine0/table_act.vio ./machine0/mess_act.vio
./machine0/del.vo ./machine0/del.glob ./machine0/del.v.beautified: ./machine0/del.v ./machine0/init.vo ./machine0/table_act.vo ./machine0/mess_act.vo
./machine0/del.vio: ./machine0/del.v ./machine0/init.vio ./machine0/table_act.vio ./machine0/mess_act.vio
./machine0/rece_copy2.vo ./machine0/rece_copy2.glob ./machine0/rece_copy2.v.beautified: ./machine0/rece_copy2.v ./machine0/init.vo ./machine0/table_act.vo ./machine0/mess_act.vo
./machine0/rece_copy2.vio: ./machine0/rece_copy2.v ./machine0/init.vio ./machine0/table_act.vio ./machine0/mess_act.vio
./machine0/mess_act.vo ./machine0/mess_act.glob ./machine0/mess_act.v.beautified: ./machine0/mess_act.v ./machine0/counting.vo
./machine0/mess_act.vio: ./machine0/mess_act.v ./machine0/counting.vio
./machine0/machine.vo ./machine0/machine.glob ./machine0/machine.v.beautified: ./machine0/machine.v ./abstract/fifo.vo ./abstract/table.vo
./machine0/machine.vio: ./machine0/machine.v ./abstract/fifo.vio ./abstract/table.vio
./machine0/evol.vo ./machine0/evol.glob ./machine0/evol.v.beautified: ./machine0/evol.v ./machine0/copy.vo ./machine0/del.vo ./machine0/rece_copy1.vo ./machine0/rece_copy2.vo ./machine0/rece_copy3.vo ./machine0/rece_dec.vo ./machine0/rece_inc.vo
./machine0/evol.vio: ./machine0/evol.v ./machine0/copy.vio ./machine0/del.vio ./machine0/rece_copy1.vio ./machine0/rece_copy2.vio ./machine0/rece_copy3.vio ./machine0/rece_dec.vio ./machine0/rece_inc.vio
./machine1/invariant4.vo ./machine1/invariant4.glob ./machine1/invariant4.v.beautified: ./machine1/invariant4.v ./machine1/invariant0.vo ./machine1/invariant1.vo ./machine1/invariant2.vo ./machine1/invariant3.vo
./machine1/invariant4.vio: ./machine1/invariant4.v ./machine1/invariant0.vio ./machine1/invariant1.vio ./machine1/invariant2.vio ./machine1/invariant3.vio
./machine1/invariant2.vo ./machine1/invariant2.glob ./machine1/invariant2.v.beautified: ./machine1/invariant2.v ./machine1/machine.vo ./machine1/cardinal.vo ./machine1/comm.vo
./machine1/invariant2.vio: ./machine1/invariant2.v ./machine1/machine.vio ./machine1/cardinal.vio ./machine1/comm.vio
./machine1/invariant8.vo ./machine1/invariant8.glob ./machine1/invariant8.v.beautified: ./machine1/invariant8.v ./machine1/invariant5.vo ./machine1/invariant7.vo
./machine1/invariant8.vio: ./machine1/invariant8.v ./machine1/invariant5.vio ./machine1/invariant7.vio
./machine1/cardinal.vo ./machine1/cardinal.glob ./machine1/cardinal.v.beautified: ./machine1/cardinal.v ./abstract/reduce.vo ./machine1/machine.vo
./machine1/cardinal.vio: ./machine1/cardinal.v ./abstract/reduce.vio ./machine1/machine.vio
./machine1/invariant6.vo ./machine1/invariant6.glob ./machine1/invariant6.v.beautified: ./machine1/invariant6.v ./machine1/invariant0.vo ./machine1/invariant1.vo ./machine1/invariant2.vo ./machine1/invariant3.vo ./machine1/invariant4.vo
./machine1/invariant6.vio: ./machine1/invariant6.v ./machine1/invariant0.vio ./machine1/invariant1.vio ./machine1/invariant2.vio ./machine1/invariant3.vio ./machine1/invariant4.vio
./machine1/invariant3.vo ./machine1/invariant3.glob ./machine1/invariant3.v.beautified: ./machine1/invariant3.v ./machine1/invariant0.vo ./machine1/invariant1.vo ./machine1/invariant2.vo
./machine1/invariant3.vio: ./machine1/invariant3.v ./machine1/invariant0.vio ./machine1/invariant1.vio ./machine1/invariant2.vio
./machine1/alternate.vo ./machine1/alternate.glob ./machine1/alternate.v.beautified: ./machine1/alternate.v ./machine1/machine.vo ./machine1/cardinal.vo ./machine1/comm.vo
./machine1/alternate.vio: ./machine1/alternate.v ./machine1/machine.vio ./machine1/cardinal.vio ./machine1/comm.vio
./machine1/comm.vo ./machine1/comm.glob ./machine1/comm.v.beautified: ./machine1/comm.v ./abstract/reduce.vo ./machine1/cardinal.vo ./abstract/sigma2.vo ./abstract/sum.vo
./machine1/comm.vio: ./machine1/comm.v ./abstract/reduce.vio ./machine1/cardinal.vio ./abstract/sigma2.vio ./abstract/sum.vio
./machine1/invariant7.vo ./machine1/invariant7.glob ./machine1/invariant7.v.beautified: ./machine1/invariant7.v ./machine1/invariant6.vo
./machine1/invariant7.vio: ./machine1/invariant7.v ./machine1/invariant6.vio
./machine1/invariant0.vo ./machine1/invariant0.glob ./machine1/invariant0.v.beautified: ./machine1/invariant0.v ./machine1/machine.vo ./machine1/cardinal.vo ./machine1/comm.vo
./machine1/invariant0.vio: ./machine1/invariant0.v ./machine1/machine.vio ./machine1/cardinal.vio ./machine1/comm.vio
./machine1/invariant1.vo ./machine1/invariant1.glob ./machine1/invariant1.v.beautified: ./machine1/invariant1.v ./machine1/machine.vo ./machine1/cardinal.vo ./machine1/comm.vo
./machine1/invariant1.vio: ./machine1/invariant1.v ./machine1/machine.vio ./machine1/cardinal.vio ./machine1/comm.vio
./machine1/machine.vo ./machine1/machine.glob ./machine1/machine.v.beautified: ./machine1/machine.v ./abstract/abstract_machine.vo
./machine1/machine.vio: ./machine1/machine.v ./abstract/abstract_machine.vio
./machine1/invariant5.vo ./machine1/invariant5.glob ./machine1/invariant5.v.beautified: ./machine1/invariant5.v ./machine1/alternate.vo ./machine1/invariant0.vo ./machine1/invariant1.vo ./machine1/invariant2.vo ./machine1/invariant3.vo ./machine1/invariant4.vo
./machine1/invariant5.vio: ./machine1/invariant5.v ./machine1/alternate.vio ./machine1/invariant0.vio ./machine1/invariant1.vio ./machine1/invariant2.vio ./machine1/invariant3.vio ./machine1/invariant4.vio
./machine4/invariant4.vo ./machine4/invariant4.glob ./machine4/invariant4.v.beautified: ./machine4/invariant4.v ./machine4/invariant0.vo ./machine4/invariant1.vo ./machine4/invariant2.vo ./machine4/invariant3.vo
./machine4/invariant4.vio: ./machine4/invariant4.v ./machine4/invariant0.vio ./machine4/invariant1.vio ./machine4/invariant2.vio ./machine4/invariant3.vio
./machine4/invariant2.vo ./machine4/invariant2.glob ./machine4/invariant2.v.beautified: ./machine4/invariant2.v ./machine4/machine.vo ./machine4/cardinal.vo ./machine4/comm.vo
./machine4/invariant2.vio: ./machine4/invariant2.v ./machine4/machine.vio ./machine4/cardinal.vio ./machine4/comm.vio
./machine4/invariant8.vo ./machine4/invariant8.glob ./machine4/invariant8.v.beautified: ./machine4/invariant8.v ./machine4/invariant5.vo ./machine4/invariant7.vo
./machine4/invariant8.vio: ./machine4/invariant8.v ./machine4/invariant5.vio ./machine4/invariant7.vio
./machine4/cardinal.vo ./machine4/cardinal.glob ./machine4/cardinal.v.beautified: ./machine4/cardinal.v ./abstract/reduce.vo ./machine4/machine.vo
./machine4/cardinal.vio: ./machine4/cardinal.v ./abstract/reduce.vio ./machine4/machine.vio
./machine4/invariant6.vo ./machine4/invariant6.glob ./machine4/invariant6.v.beautified: ./machine4/invariant6.v ./machine4/invariant0.vo ./machine4/invariant1.vo ./machine4/invariant2.vo ./machine4/invariant3.vo ./machine4/invariant4.vo
./machine4/invariant6.vio: ./machine4/invariant6.v ./machine4/invariant0.vio ./machine4/invariant1.vio ./machine4/invariant2.vio ./machine4/invariant3.vio ./machine4/invariant4.vio
./machine4/invariant3.vo ./machine4/invariant3.glob ./machine4/invariant3.v.beautified: ./machine4/invariant3.v ./machine4/invariant0.vo ./machine4/invariant1.vo ./machine4/invariant2.vo
./machine4/invariant3.vio: ./machine4/invariant3.v ./machine4/invariant0.vio ./machine4/invariant1.vio ./machine4/invariant2.vio
./machine4/alternate.vo ./machine4/alternate.glob ./machine4/alternate.v.beautified: ./machine4/alternate.v ./machine4/machine.vo ./machine4/cardinal.vo ./machine4/comm.vo
./machine4/alternate.vio: ./machine4/alternate.v ./machine4/machine.vio ./machine4/cardinal.vio ./machine4/comm.vio
./machine4/comm.vo ./machine4/comm.glob ./machine4/comm.v.beautified: ./machine4/comm.v ./abstract/reduce.vo ./machine4/cardinal.vo ./abstract/sigma2.vo ./abstract/sum.vo
./machine4/comm.vio: ./machine4/comm.v ./abstract/reduce.vio ./machine4/cardinal.vio ./abstract/sigma2.vio ./abstract/sum.vio
./machine4/invariant7.vo ./machine4/invariant7.glob ./machine4/invariant7.v.beautified: ./machine4/invariant7.v ./machine4/invariant6.vo
./machine4/invariant7.vio: ./machine4/invariant7.v ./machine4/invariant6.vio
./machine4/invariant0.vo ./machine4/invariant0.glob ./machine4/invariant0.v.beautified: ./machine4/invariant0.v ./machine4/machine.vo ./machine4/cardinal.vo ./machine4/comm.vo
./machine4/invariant0.vio: ./machine4/invariant0.v ./machine4/machine.vio ./machine4/cardinal.vio ./machine4/comm.vio
./machine4/invariant1.vo ./machine4/invariant1.glob ./machine4/invariant1.v.beautified: ./machine4/invariant1.v ./machine4/machine.vo ./machine4/cardinal.vo ./machine4/comm.vo
./machine4/invariant1.vio: ./machine4/invariant1.v ./machine4/machine.vio ./machine4/cardinal.vio ./machine4/comm.vio
./machine4/machine.vo ./machine4/machine.glob ./machine4/machine.v.beautified: ./machine4/machine.v ./abstract/abstract_machine.vo
./machine4/machine.vio: ./machine4/machine.v ./abstract/abstract_machine.vio
./machine4/invariant5.vo ./machine4/invariant5.glob ./machine4/invariant5.v.beautified: ./machine4/invariant5.v ./machine4/alternate.vo ./machine4/invariant0.vo ./machine4/invariant1.vo ./machine4/invariant2.vo ./machine4/invariant3.vo ./machine4/invariant4.vo
./machine4/invariant5.vio: ./machine4/invariant5.v ./machine4/alternate.vio ./machine4/invariant0.vio ./machine4/invariant1.vio ./machine4/invariant2.vio ./machine4/invariant3.vio ./machine4/invariant4.vio

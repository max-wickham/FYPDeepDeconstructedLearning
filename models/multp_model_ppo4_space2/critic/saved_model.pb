��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*
2.12.0-rc12v2.12.0-rc0-46-g0d8efc960d28��
r
dense_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_35/bias
k
!dense_35/bias/Read/ReadVariableOpReadVariableOpdense_35/bias*
_output_shapes
:*
dtype0
z
dense_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_35/kernel
s
#dense_35/kernel/Read/ReadVariableOpReadVariableOpdense_35/kernel*
_output_shapes

:*
dtype0
r
dense_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_34/bias
k
!dense_34/bias/Read/ReadVariableOpReadVariableOpdense_34/bias*
_output_shapes
:*
dtype0
z
dense_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_34/kernel
s
#dense_34/kernel/Read/ReadVariableOpReadVariableOpdense_34/kernel*
_output_shapes

:
*
dtype0
r
dense_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_33/bias
k
!dense_33/bias/Read/ReadVariableOpReadVariableOpdense_33/bias*
_output_shapes
:
*
dtype0
|
dense_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�
* 
shared_namedense_33/kernel
u
#dense_33/kernel/Read/ReadVariableOpReadVariableOpdense_33/kernel* 
_output_shapes
:
�
*
dtype0
~
serving_default_input_4Placeholder*)
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4dense_33/kerneldense_33/biasdense_34/kerneldense_34/biasdense_35/kerneldense_35/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference_signature_wrapper_12540

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias*
.
0
1
2
3
#4
$5*
.
0
1
2
3
#4
$5*
* 
�
%non_trainable_variables

&layers
'metrics
(layer_regularization_losses
)layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
*trace_0
+trace_1
,trace_2
-trace_3* 
6
.trace_0
/trace_1
0trace_2
1trace_3* 
* 

2serving_default* 

0
1*

0
1*
* 
�
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

8trace_0* 

9trace_0* 
_Y
VARIABLE_VALUEdense_33/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_33/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

?trace_0* 

@trace_0* 
_Y
VARIABLE_VALUEdense_34/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_34/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

#0
$1*

#0
$1*
* 
�
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

Ftrace_0* 

Gtrace_0* 
_Y
VARIABLE_VALUEdense_35/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_35/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_33/kerneldense_33/biasdense_34/kerneldense_34/biasdense_35/kerneldense_35/biasConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference__traced_save_12740
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_33/kerneldense_33/biasdense_34/kerneldense_34/biasdense_35/kerneldense_35/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_restore_12768��
�
�
B__inference_model_3_layer_call_and_return_conditional_losses_12365
input_4"
dense_33_12326:
�

dense_33_12328:
 
dense_34_12343:

dense_34_12345: 
dense_35_12359:
dense_35_12361:
identity�� dense_33/StatefulPartitionedCall� dense_34/StatefulPartitionedCall� dense_35/StatefulPartitionedCall�
 dense_33/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_33_12326dense_33_12328*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_33_layer_call_and_return_conditional_losses_12325�
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0dense_34_12343dense_34_12345*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_34_layer_call_and_return_conditional_losses_12342�
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_12359dense_35_12361*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_35_layer_call_and_return_conditional_losses_12358x
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:����������: : : : : : 2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:R N
)
_output_shapes
:����������
!
_user_specified_name	input_4
�	
�
C__inference_dense_35_layer_call_and_return_conditional_losses_12681

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
C__inference_dense_34_layer_call_and_return_conditional_losses_12662

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�

�
C__inference_dense_33_layer_call_and_return_conditional_losses_12642

inputs2
matmul_readvariableop_resource:
�
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
B__inference_model_3_layer_call_and_return_conditional_losses_12384
input_4"
dense_33_12368:
�

dense_33_12370:
 
dense_34_12373:

dense_34_12375: 
dense_35_12378:
dense_35_12380:
identity�� dense_33/StatefulPartitionedCall� dense_34/StatefulPartitionedCall� dense_35/StatefulPartitionedCall�
 dense_33/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_33_12368dense_33_12370*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_33_layer_call_and_return_conditional_losses_12325�
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0dense_34_12373dense_34_12375*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_34_layer_call_and_return_conditional_losses_12342�
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_12378dense_35_12380*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_35_layer_call_and_return_conditional_losses_12358x
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:����������: : : : : : 2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:R N
)
_output_shapes
:����������
!
_user_specified_name	input_4
�

�
C__inference_dense_34_layer_call_and_return_conditional_losses_12342

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
'__inference_model_3_layer_call_fn_12457
input_4
unknown:
�

	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_12442o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
)
_output_shapes
:����������
!
_user_specified_name	input_4
�
�
B__inference_model_3_layer_call_and_return_conditional_losses_12442

inputs"
dense_33_12426:
�

dense_33_12428:
 
dense_34_12431:

dense_34_12433: 
dense_35_12436:
dense_35_12438:
identity�� dense_33/StatefulPartitionedCall� dense_34/StatefulPartitionedCall� dense_35/StatefulPartitionedCall�
 dense_33/StatefulPartitionedCallStatefulPartitionedCallinputsdense_33_12426dense_33_12428*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_33_layer_call_and_return_conditional_losses_12325�
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0dense_34_12431dense_34_12433*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_34_layer_call_and_return_conditional_losses_12342�
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_12436dense_35_12438*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_35_layer_call_and_return_conditional_losses_12358x
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:����������: : : : : : 2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:Q M
)
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_dense_34_layer_call_fn_12651

inputs
unknown:

	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_34_layer_call_and_return_conditional_losses_12342o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
'__inference_model_3_layer_call_fn_12421
input_4
unknown:
�

	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_12406o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
)
_output_shapes
:����������
!
_user_specified_name	input_4
�	
�
C__inference_dense_35_layer_call_and_return_conditional_losses_12358

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_dense_35_layer_call_fn_12671

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_35_layer_call_and_return_conditional_losses_12358o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
 __inference__wrapped_model_12310
input_4C
/model_3_dense_33_matmul_readvariableop_resource:
�
>
0model_3_dense_33_biasadd_readvariableop_resource:
A
/model_3_dense_34_matmul_readvariableop_resource:
>
0model_3_dense_34_biasadd_readvariableop_resource:A
/model_3_dense_35_matmul_readvariableop_resource:>
0model_3_dense_35_biasadd_readvariableop_resource:
identity��'model_3/dense_33/BiasAdd/ReadVariableOp�&model_3/dense_33/MatMul/ReadVariableOp�'model_3/dense_34/BiasAdd/ReadVariableOp�&model_3/dense_34/MatMul/ReadVariableOp�'model_3/dense_35/BiasAdd/ReadVariableOp�&model_3/dense_35/MatMul/ReadVariableOp�
&model_3/dense_33/MatMul/ReadVariableOpReadVariableOp/model_3_dense_33_matmul_readvariableop_resource* 
_output_shapes
:
�
*
dtype0�
model_3/dense_33/MatMulMatMulinput_4.model_3/dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
'model_3/dense_33/BiasAdd/ReadVariableOpReadVariableOp0model_3_dense_33_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
model_3/dense_33/BiasAddBiasAdd!model_3/dense_33/MatMul:product:0/model_3/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
model_3/dense_33/ReluRelu!model_3/dense_33/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
&model_3/dense_34/MatMul/ReadVariableOpReadVariableOp/model_3_dense_34_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
model_3/dense_34/MatMulMatMul#model_3/dense_33/Relu:activations:0.model_3/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'model_3/dense_34/BiasAdd/ReadVariableOpReadVariableOp0model_3_dense_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_3/dense_34/BiasAddBiasAdd!model_3/dense_34/MatMul:product:0/model_3/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
model_3/dense_34/ReluRelu!model_3/dense_34/BiasAdd:output:0*
T0*'
_output_shapes
:����������
&model_3/dense_35/MatMul/ReadVariableOpReadVariableOp/model_3_dense_35_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
model_3/dense_35/MatMulMatMul#model_3/dense_34/Relu:activations:0.model_3/dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'model_3/dense_35/BiasAdd/ReadVariableOpReadVariableOp0model_3_dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_3/dense_35/BiasAddBiasAdd!model_3/dense_35/MatMul:product:0/model_3/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������p
IdentityIdentity!model_3/dense_35/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp(^model_3/dense_33/BiasAdd/ReadVariableOp'^model_3/dense_33/MatMul/ReadVariableOp(^model_3/dense_34/BiasAdd/ReadVariableOp'^model_3/dense_34/MatMul/ReadVariableOp(^model_3/dense_35/BiasAdd/ReadVariableOp'^model_3/dense_35/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:����������: : : : : : 2R
'model_3/dense_33/BiasAdd/ReadVariableOp'model_3/dense_33/BiasAdd/ReadVariableOp2P
&model_3/dense_33/MatMul/ReadVariableOp&model_3/dense_33/MatMul/ReadVariableOp2R
'model_3/dense_34/BiasAdd/ReadVariableOp'model_3/dense_34/BiasAdd/ReadVariableOp2P
&model_3/dense_34/MatMul/ReadVariableOp&model_3/dense_34/MatMul/ReadVariableOp2R
'model_3/dense_35/BiasAdd/ReadVariableOp'model_3/dense_35/BiasAdd/ReadVariableOp2P
&model_3/dense_35/MatMul/ReadVariableOp&model_3/dense_35/MatMul/ReadVariableOp:R N
)
_output_shapes
:����������
!
_user_specified_name	input_4
�
�
B__inference_model_3_layer_call_and_return_conditional_losses_12406

inputs"
dense_33_12390:
�

dense_33_12392:
 
dense_34_12395:

dense_34_12397: 
dense_35_12400:
dense_35_12402:
identity�� dense_33/StatefulPartitionedCall� dense_34/StatefulPartitionedCall� dense_35/StatefulPartitionedCall�
 dense_33/StatefulPartitionedCallStatefulPartitionedCallinputsdense_33_12390dense_33_12392*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_33_layer_call_and_return_conditional_losses_12325�
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0dense_34_12395dense_34_12397*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_34_layer_call_and_return_conditional_losses_12342�
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_12400dense_35_12402*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_35_layer_call_and_return_conditional_losses_12358x
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:����������: : : : : : 2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:Q M
)
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_12540
input_4
unknown:
�

	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__wrapped_model_12310o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
)
_output_shapes
:����������
!
_user_specified_name	input_4
�
�
'__inference_model_3_layer_call_fn_12574

inputs
unknown:
�

	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_12442o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
!__inference__traced_restore_12768
file_prefix4
 assignvariableop_dense_33_kernel:
�
.
 assignvariableop_1_dense_33_bias:
4
"assignvariableop_2_dense_34_kernel:
.
 assignvariableop_3_dense_34_bias:4
"assignvariableop_4_dense_35_kernel:.
 assignvariableop_5_dense_35_bias:

identity_7��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH~
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_dense_33_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_33_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_34_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_34_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_35_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_35_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_7IdentityIdentity_6:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 "!

identity_7Identity_7:output:0*!
_input_shapes
: : : : : : : 2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�9
�
__inference__traced_save_12740
file_prefix:
&read_disablecopyonread_dense_33_kernel:
�
4
&read_1_disablecopyonread_dense_33_bias:
:
(read_2_disablecopyonread_dense_34_kernel:
4
&read_3_disablecopyonread_dense_34_bias::
(read_4_disablecopyonread_dense_35_kernel:4
&read_5_disablecopyonread_dense_35_bias:
savev2_const
identity_13��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_dense_33_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_dense_33_kernel^Read/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
�
*
dtype0k
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
�
c

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0* 
_output_shapes
:
�
z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_dense_33_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_dense_33_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:
*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:
_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:
|
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_dense_34_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_dense_34_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:
*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:
c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:
z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_dense_34_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_dense_34_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_dense_35_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_dense_35_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_dense_35_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_dense_35_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH{
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
	2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_12Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_13IdentityIdentity_12:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_13Identity_13:output:0*#
_input_shapes
: : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp:

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
B__inference_model_3_layer_call_and_return_conditional_losses_12598

inputs;
'dense_33_matmul_readvariableop_resource:
�
6
(dense_33_biasadd_readvariableop_resource:
9
'dense_34_matmul_readvariableop_resource:
6
(dense_34_biasadd_readvariableop_resource:9
'dense_35_matmul_readvariableop_resource:6
(dense_35_biasadd_readvariableop_resource:
identity��dense_33/BiasAdd/ReadVariableOp�dense_33/MatMul/ReadVariableOp�dense_34/BiasAdd/ReadVariableOp�dense_34/MatMul/ReadVariableOp�dense_35/BiasAdd/ReadVariableOp�dense_35/MatMul/ReadVariableOp�
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource* 
_output_shapes
:
�
*
dtype0{
dense_33/MatMulMatMulinputs&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
b
dense_33/ReluReludense_33/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
dense_34/MatMulMatMuldense_33/Relu:activations:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_34/ReluReludense_34/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_35/MatMulMatMuldense_34/Relu:activations:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_35/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:����������: : : : : : 2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp:Q M
)
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_model_3_layer_call_fn_12557

inputs
unknown:
�

	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_12406o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
B__inference_model_3_layer_call_and_return_conditional_losses_12622

inputs;
'dense_33_matmul_readvariableop_resource:
�
6
(dense_33_biasadd_readvariableop_resource:
9
'dense_34_matmul_readvariableop_resource:
6
(dense_34_biasadd_readvariableop_resource:9
'dense_35_matmul_readvariableop_resource:6
(dense_35_biasadd_readvariableop_resource:
identity��dense_33/BiasAdd/ReadVariableOp�dense_33/MatMul/ReadVariableOp�dense_34/BiasAdd/ReadVariableOp�dense_34/MatMul/ReadVariableOp�dense_35/BiasAdd/ReadVariableOp�dense_35/MatMul/ReadVariableOp�
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource* 
_output_shapes
:
�
*
dtype0{
dense_33/MatMulMatMulinputs&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
b
dense_33/ReluReludense_33/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0�
dense_34/MatMulMatMuldense_33/Relu:activations:0&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense_34/ReluReludense_34/BiasAdd:output:0*
T0*'
_output_shapes
:����������
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
dense_35/MatMulMatMuldense_34/Relu:activations:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_35/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp ^dense_34/BiasAdd/ReadVariableOp^dense_34/MatMul/ReadVariableOp ^dense_35/BiasAdd/ReadVariableOp^dense_35/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:����������: : : : : : 2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp2B
dense_34/BiasAdd/ReadVariableOpdense_34/BiasAdd/ReadVariableOp2@
dense_34/MatMul/ReadVariableOpdense_34/MatMul/ReadVariableOp2B
dense_35/BiasAdd/ReadVariableOpdense_35/BiasAdd/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp:Q M
)
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
C__inference_dense_33_layer_call_and_return_conditional_losses_12325

inputs2
matmul_readvariableop_resource:
�
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_dense_33_layer_call_fn_12631

inputs
unknown:
�

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_33_layer_call_and_return_conditional_losses_12325o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:����������
 
_user_specified_nameinputs"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
=
input_42
serving_default_input_4:0����������<
dense_350
StatefulPartitionedCall:0���������tensorflow/serving/predict:�b
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
J
0
1
2
3
#4
$5"
trackable_list_wrapper
J
0
1
2
3
#4
$5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
%non_trainable_variables

&layers
'metrics
(layer_regularization_losses
)layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
�
*trace_0
+trace_1
,trace_2
-trace_32�
'__inference_model_3_layer_call_fn_12421
'__inference_model_3_layer_call_fn_12457
'__inference_model_3_layer_call_fn_12557
'__inference_model_3_layer_call_fn_12574�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z*trace_0z+trace_1z,trace_2z-trace_3
�
.trace_0
/trace_1
0trace_2
1trace_32�
B__inference_model_3_layer_call_and_return_conditional_losses_12365
B__inference_model_3_layer_call_and_return_conditional_losses_12384
B__inference_model_3_layer_call_and_return_conditional_losses_12598
B__inference_model_3_layer_call_and_return_conditional_losses_12622�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z.trace_0z/trace_1z0trace_2z1trace_3
�B�
 __inference__wrapped_model_12310input_4"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
2serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
8trace_02�
(__inference_dense_33_layer_call_fn_12631�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z8trace_0
�
9trace_02�
C__inference_dense_33_layer_call_and_return_conditional_losses_12642�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z9trace_0
#:!
�
2dense_33/kernel
:
2dense_33/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
?trace_02�
(__inference_dense_34_layer_call_fn_12651�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z?trace_0
�
@trace_02�
C__inference_dense_34_layer_call_and_return_conditional_losses_12662�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z@trace_0
!:
2dense_34/kernel
:2dense_34/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
�
Ftrace_02�
(__inference_dense_35_layer_call_fn_12671�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zFtrace_0
�
Gtrace_02�
C__inference_dense_35_layer_call_and_return_conditional_losses_12681�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zGtrace_0
!:2dense_35/kernel
:2dense_35/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_model_3_layer_call_fn_12421input_4"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_model_3_layer_call_fn_12457input_4"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_model_3_layer_call_fn_12557inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_model_3_layer_call_fn_12574inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_model_3_layer_call_and_return_conditional_losses_12365input_4"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_model_3_layer_call_and_return_conditional_losses_12384input_4"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_model_3_layer_call_and_return_conditional_losses_12598inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_model_3_layer_call_and_return_conditional_losses_12622inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_signature_wrapper_12540input_4"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_33_layer_call_fn_12631inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_33_layer_call_and_return_conditional_losses_12642inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_34_layer_call_fn_12651inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_34_layer_call_and_return_conditional_losses_12662inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_35_layer_call_fn_12671inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_35_layer_call_and_return_conditional_losses_12681inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
 __inference__wrapped_model_12310q#$2�/
(�%
#� 
input_4����������
� "3�0
.
dense_35"�
dense_35����������
C__inference_dense_33_layer_call_and_return_conditional_losses_12642e1�.
'�$
"�
inputs����������
� ",�)
"�
tensor_0���������

� �
(__inference_dense_33_layer_call_fn_12631Z1�.
'�$
"�
inputs����������
� "!�
unknown���������
�
C__inference_dense_34_layer_call_and_return_conditional_losses_12662c/�,
%�"
 �
inputs���������

� ",�)
"�
tensor_0���������
� �
(__inference_dense_34_layer_call_fn_12651X/�,
%�"
 �
inputs���������

� "!�
unknown����������
C__inference_dense_35_layer_call_and_return_conditional_losses_12681c#$/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
(__inference_dense_35_layer_call_fn_12671X#$/�,
%�"
 �
inputs���������
� "!�
unknown����������
B__inference_model_3_layer_call_and_return_conditional_losses_12365r#$:�7
0�-
#� 
input_4����������
p

 
� ",�)
"�
tensor_0���������
� �
B__inference_model_3_layer_call_and_return_conditional_losses_12384r#$:�7
0�-
#� 
input_4����������
p 

 
� ",�)
"�
tensor_0���������
� �
B__inference_model_3_layer_call_and_return_conditional_losses_12598q#$9�6
/�,
"�
inputs����������
p

 
� ",�)
"�
tensor_0���������
� �
B__inference_model_3_layer_call_and_return_conditional_losses_12622q#$9�6
/�,
"�
inputs����������
p 

 
� ",�)
"�
tensor_0���������
� �
'__inference_model_3_layer_call_fn_12421g#$:�7
0�-
#� 
input_4����������
p

 
� "!�
unknown����������
'__inference_model_3_layer_call_fn_12457g#$:�7
0�-
#� 
input_4����������
p 

 
� "!�
unknown����������
'__inference_model_3_layer_call_fn_12557f#$9�6
/�,
"�
inputs����������
p

 
� "!�
unknown����������
'__inference_model_3_layer_call_fn_12574f#$9�6
/�,
"�
inputs����������
p 

 
� "!�
unknown����������
#__inference_signature_wrapper_12540|#$=�:
� 
3�0
.
input_4#� 
input_4����������"3�0
.
dense_35"�
dense_35���������
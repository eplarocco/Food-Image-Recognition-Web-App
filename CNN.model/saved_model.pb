Î§
ä
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
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
executor_typestring ¨
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Óû

conv2d_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_22/kernel
}
$conv2d_22/kernel/Read/ReadVariableOpReadVariableOpconv2d_22/kernel*&
_output_shapes
:*
dtype0
t
conv2d_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_22/bias
m
"conv2d_22/bias/Read/ReadVariableOpReadVariableOpconv2d_22/bias*
_output_shapes
:*
dtype0

conv2d_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_23/kernel
}
$conv2d_23/kernel/Read/ReadVariableOpReadVariableOpconv2d_23/kernel*&
_output_shapes
: *
dtype0
t
conv2d_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_23/bias
m
"conv2d_23/bias/Read/ReadVariableOpReadVariableOpconv2d_23/bias*
_output_shapes
: *
dtype0

conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_24/kernel
}
$conv2d_24/kernel/Read/ReadVariableOpReadVariableOpconv2d_24/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_24/bias
m
"conv2d_24/bias/Read/ReadVariableOpReadVariableOpconv2d_24/bias*
_output_shapes
:@*
dtype0

conv2d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_25/kernel
}
$conv2d_25/kernel/Read/ReadVariableOpReadVariableOpconv2d_25/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_25/bias
m
"conv2d_25/bias/Read/ReadVariableOpReadVariableOpconv2d_25/bias*
_output_shapes
:@*
dtype0
|
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
À(* 
shared_namedense_12/kernel
u
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel* 
_output_shapes
:
À(*
dtype0
s
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_12/bias
l
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes	
:*
dtype0
{
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_namedense_13/kernel
t
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes
:	*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:*
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

random_flip_6/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*'
shared_namerandom_flip_6/StateVar
}
*random_flip_6/StateVar/Read/ReadVariableOpReadVariableOprandom_flip_6/StateVar*
_output_shapes
:*
dtype0	

random_rotation_6/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*+
shared_namerandom_rotation_6/StateVar

.random_rotation_6/StateVar/Read/ReadVariableOpReadVariableOprandom_rotation_6/StateVar*
_output_shapes
:*
dtype0	

random_zoom_6/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*'
shared_namerandom_zoom_6/StateVar
}
*random_zoom_6/StateVar/Read/ReadVariableOpReadVariableOprandom_zoom_6/StateVar*
_output_shapes
:*
dtype0	

RMSprop/conv2d_22/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameRMSprop/conv2d_22/kernel/rms

0RMSprop/conv2d_22/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_22/kernel/rms*&
_output_shapes
:*
dtype0

RMSprop/conv2d_22/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameRMSprop/conv2d_22/bias/rms

.RMSprop/conv2d_22/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_22/bias/rms*
_output_shapes
:*
dtype0

RMSprop/conv2d_23/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameRMSprop/conv2d_23/kernel/rms

0RMSprop/conv2d_23/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_23/kernel/rms*&
_output_shapes
: *
dtype0

RMSprop/conv2d_23/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameRMSprop/conv2d_23/bias/rms

.RMSprop/conv2d_23/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_23/bias/rms*
_output_shapes
: *
dtype0

RMSprop/conv2d_24/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*-
shared_nameRMSprop/conv2d_24/kernel/rms

0RMSprop/conv2d_24/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_24/kernel/rms*&
_output_shapes
: @*
dtype0

RMSprop/conv2d_24/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameRMSprop/conv2d_24/bias/rms

.RMSprop/conv2d_24/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_24/bias/rms*
_output_shapes
:@*
dtype0

RMSprop/conv2d_25/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*-
shared_nameRMSprop/conv2d_25/kernel/rms

0RMSprop/conv2d_25/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_25/kernel/rms*&
_output_shapes
:@@*
dtype0

RMSprop/conv2d_25/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_nameRMSprop/conv2d_25/bias/rms

.RMSprop/conv2d_25/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_25/bias/rms*
_output_shapes
:@*
dtype0

RMSprop/dense_12/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
À(*,
shared_nameRMSprop/dense_12/kernel/rms

/RMSprop/dense_12/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_12/kernel/rms* 
_output_shapes
:
À(*
dtype0

RMSprop/dense_12/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameRMSprop/dense_12/bias/rms

-RMSprop/dense_12/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_12/bias/rms*
_output_shapes	
:*
dtype0

RMSprop/dense_13/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_nameRMSprop/dense_13/kernel/rms

/RMSprop/dense_13/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_13/kernel/rms*
_output_shapes
:	*
dtype0

RMSprop/dense_13/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameRMSprop/dense_13/bias/rms

-RMSprop/dense_13/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_13/bias/rms*
_output_shapes
:*
dtype0

NoOpNoOp
¦l
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ák
value×kBÔk BÍk

layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
·
layer-0
layer-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses* 
¦

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses*

.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses* 
¦

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses*

<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses* 
¦

Bkernel
Cbias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses*

J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses* 
¦

Pkernel
Qbias
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses*

X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses* 

^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses* 
¦

dkernel
ebias
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses*
¦

lkernel
mbias
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses*
Ó
titer
	udecay
vlearning_rate
wmomentum
xrho
&rmsõ
'rmsö
4rms÷
5rmsø
Brmsù
Crmsú
Prmsû
Qrmsü
drmsý
ermsþ
lrmsÿ
mrms*
Z
&0
'1
42
53
B4
C5
P6
Q7
d8
e9
l10
m11*
Z
&0
'1
42
53
B4
C5
P6
Q7
d8
e9
l10
m11*
* 
°
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

~serving_default* 
­
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses*
®
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses*
®
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEconv2d_22/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_22/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

&0
'1*

&0
'1*
* 

non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEconv2d_23/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_23/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

40
51*

40
51*
* 

¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEconv2d_24/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_24/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

B0
C1*

B0
C1*
* 

²non_trainable_variables
³layers
´metrics
 µlayer_regularization_losses
¶layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses* 
* 
* 
`Z
VARIABLE_VALUEconv2d_25/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_25/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

P0
Q1*

P0
Q1*
* 

¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ánon_trainable_variables
Âlayers
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_12/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

d0
e1*

d0
e1*
* 

Ënon_trainable_variables
Ìlayers
Ímetrics
 Îlayer_regularization_losses
Ïlayer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_13/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

l0
m1*

l0
m1*
* 

Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses*
* 
* 
OI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE*
* 
b
0
1
2
3
4
5
6
7
	8

9
10
11
12*

Õ0
Ö1*
* 
* 
* 
* 
* 
* 

×non_trainable_variables
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ûlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

Ü
_generator*
* 
* 
* 
* 
* 

Ýnon_trainable_variables
Þlayers
ßmetrics
 àlayer_regularization_losses
álayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

â
_generator*
* 
* 
* 
* 
* 

ãnon_trainable_variables
älayers
åmetrics
 ælayer_regularization_losses
çlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

è
_generator*
* 
* 
* 

0
1
2*
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
<

étotal

êcount
ë	variables
ì	keras_api*
M

ítotal

îcount
ï
_fn_kwargs
ð	variables
ñ	keras_api*
* 
* 
* 
* 
* 

ò
_state_var*
* 
* 
* 
* 
* 

ó
_state_var*
* 
* 
* 
* 
* 

ô
_state_var*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

é0
ê1*

ë	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

í0
î1*

ð	variables*
|
VARIABLE_VALUErandom_flip_6/StateVarRlayer-0/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUErandom_rotation_6/StateVarRlayer-0/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUErandom_zoom_6/StateVarRlayer-0/layer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/conv2d_22/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/conv2d_22/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/conv2d_23/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/conv2d_23/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/conv2d_24/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/conv2d_24/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/conv2d_25/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/conv2d_25/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/dense_12/kernel/rmsTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/dense_12/bias/rmsRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/dense_13/kernel/rmsTlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/dense_13/bias/rmsRlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_7Placeholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ´´

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_7conv2d_22/kernelconv2d_22/biasconv2d_23/kernelconv2d_23/biasconv2d_24/kernelconv2d_24/biasconv2d_25/kernelconv2d_25/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_46168
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ð
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_22/kernel/Read/ReadVariableOp"conv2d_22/bias/Read/ReadVariableOp$conv2d_23/kernel/Read/ReadVariableOp"conv2d_23/bias/Read/ReadVariableOp$conv2d_24/kernel/Read/ReadVariableOp"conv2d_24/bias/Read/ReadVariableOp$conv2d_25/kernel/Read/ReadVariableOp"conv2d_25/bias/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*random_flip_6/StateVar/Read/ReadVariableOp.random_rotation_6/StateVar/Read/ReadVariableOp*random_zoom_6/StateVar/Read/ReadVariableOp0RMSprop/conv2d_22/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_22/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_23/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_23/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_24/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_24/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_25/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_25/bias/rms/Read/ReadVariableOp/RMSprop/dense_12/kernel/rms/Read/ReadVariableOp-RMSprop/dense_12/bias/rms/Read/ReadVariableOp/RMSprop/dense_13/kernel/rms/Read/ReadVariableOp-RMSprop/dense_13/bias/rms/Read/ReadVariableOpConst*1
Tin*
(2&				*
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
GPU2*0J 8 *'
f"R 
__inference__traced_save_47114

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_22/kernelconv2d_22/biasconv2d_23/kernelconv2d_23/biasconv2d_24/kernelconv2d_24/biasconv2d_25/kernelconv2d_25/biasdense_12/kerneldense_12/biasdense_13/kerneldense_13/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcounttotal_1count_1random_flip_6/StateVarrandom_rotation_6/StateVarrandom_zoom_6/StateVarRMSprop/conv2d_22/kernel/rmsRMSprop/conv2d_22/bias/rmsRMSprop/conv2d_23/kernel/rmsRMSprop/conv2d_23/bias/rmsRMSprop/conv2d_24/kernel/rmsRMSprop/conv2d_24/bias/rmsRMSprop/conv2d_25/kernel/rmsRMSprop/conv2d_25/bias/rmsRMSprop/dense_12/kernel/rmsRMSprop/dense_12/bias/rmsRMSprop/dense_13/kernel/rmsRMSprop/dense_13/bias/rms*0
Tin)
'2%*
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
GPU2*0J 8 **
f%R#
!__inference__traced_restore_47232¬

ý
D__inference_conv2d_25_layer_call_and_return_conditional_losses_45306

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
æý
¯
H__inference_sequential_13_layer_call_and_return_conditional_losses_46137

inputs[
Msequential_12_random_flip_6_stateful_uniform_full_int_rngreadandskip_resource:	V
Hsequential_12_random_rotation_6_stateful_uniform_rngreadandskip_resource:	R
Dsequential_12_random_zoom_6_stateful_uniform_rngreadandskip_resource:	B
(conv2d_22_conv2d_readvariableop_resource:7
)conv2d_22_biasadd_readvariableop_resource:B
(conv2d_23_conv2d_readvariableop_resource: 7
)conv2d_23_biasadd_readvariableop_resource: B
(conv2d_24_conv2d_readvariableop_resource: @7
)conv2d_24_biasadd_readvariableop_resource:@B
(conv2d_25_conv2d_readvariableop_resource:@@7
)conv2d_25_biasadd_readvariableop_resource:@;
'dense_12_matmul_readvariableop_resource:
À(7
(dense_12_biasadd_readvariableop_resource:	:
'dense_13_matmul_readvariableop_resource:	6
(dense_13_biasadd_readvariableop_resource:
identity¢ conv2d_22/BiasAdd/ReadVariableOp¢conv2d_22/Conv2D/ReadVariableOp¢ conv2d_23/BiasAdd/ReadVariableOp¢conv2d_23/Conv2D/ReadVariableOp¢ conv2d_24/BiasAdd/ReadVariableOp¢conv2d_24/Conv2D/ReadVariableOp¢ conv2d_25/BiasAdd/ReadVariableOp¢conv2d_25/Conv2D/ReadVariableOp¢dense_12/BiasAdd/ReadVariableOp¢dense_12/MatMul/ReadVariableOp¢dense_13/BiasAdd/ReadVariableOp¢dense_13/MatMul/ReadVariableOp¢Dsequential_12/random_flip_6/stateful_uniform_full_int/RngReadAndSkip¢?sequential_12/random_rotation_6/stateful_uniform/RngReadAndSkip¢;sequential_12/random_zoom_6/stateful_uniform/RngReadAndSkip
;sequential_12/random_flip_6/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:
;sequential_12/random_flip_6/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: ï
:sequential_12/random_flip_6/stateful_uniform_full_int/ProdProdDsequential_12/random_flip_6/stateful_uniform_full_int/shape:output:0Dsequential_12/random_flip_6/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: ~
<sequential_12/random_flip_6/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :¹
<sequential_12/random_flip_6/stateful_uniform_full_int/Cast_1CastCsequential_12/random_flip_6/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Ê
Dsequential_12/random_flip_6/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkipMsequential_12_random_flip_6_stateful_uniform_full_int_rngreadandskip_resourceEsequential_12/random_flip_6/stateful_uniform_full_int/Cast/x:output:0@sequential_12/random_flip_6/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:
Isequential_12/random_flip_6/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Ksequential_12/random_flip_6/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Ksequential_12/random_flip_6/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
Csequential_12/random_flip_6/stateful_uniform_full_int/strided_sliceStridedSliceLsequential_12/random_flip_6/stateful_uniform_full_int/RngReadAndSkip:value:0Rsequential_12/random_flip_6/stateful_uniform_full_int/strided_slice/stack:output:0Tsequential_12/random_flip_6/stateful_uniform_full_int/strided_slice/stack_1:output:0Tsequential_12/random_flip_6/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskÇ
=sequential_12/random_flip_6/stateful_uniform_full_int/BitcastBitcastLsequential_12/random_flip_6/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Ksequential_12/random_flip_6/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Msequential_12/random_flip_6/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Msequential_12/random_flip_6/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
Esequential_12/random_flip_6/stateful_uniform_full_int/strided_slice_1StridedSliceLsequential_12/random_flip_6/stateful_uniform_full_int/RngReadAndSkip:value:0Tsequential_12/random_flip_6/stateful_uniform_full_int/strided_slice_1/stack:output:0Vsequential_12/random_flip_6/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Vsequential_12/random_flip_6/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Ë
?sequential_12/random_flip_6/stateful_uniform_full_int/Bitcast_1BitcastNsequential_12/random_flip_6/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0{
9sequential_12/random_flip_6/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :
5sequential_12/random_flip_6/stateful_uniform_full_intStatelessRandomUniformFullIntV2Dsequential_12/random_flip_6/stateful_uniform_full_int/shape:output:0Hsequential_12/random_flip_6/stateful_uniform_full_int/Bitcast_1:output:0Fsequential_12/random_flip_6/stateful_uniform_full_int/Bitcast:output:0Bsequential_12/random_flip_6/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	p
&sequential_12/random_flip_6/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R Ì
!sequential_12/random_flip_6/stackPack>sequential_12/random_flip_6/stateful_uniform_full_int:output:0/sequential_12/random_flip_6/zeros_like:output:0*
N*
T0	*
_output_shapes

:
/sequential_12/random_flip_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
1sequential_12/random_flip_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
1sequential_12/random_flip_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
)sequential_12/random_flip_6/strided_sliceStridedSlice*sequential_12/random_flip_6/stack:output:08sequential_12/random_flip_6/strided_slice/stack:output:0:sequential_12/random_flip_6/strided_slice/stack_1:output:0:sequential_12/random_flip_6/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maskº
Osequential_12/random_flip_6/stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´Ê
Bsequential_12/random_flip_6/stateless_random_flip_left_right/ShapeShapeXsequential_12/random_flip_6/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:
Psequential_12/random_flip_6/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Rsequential_12/random_flip_6/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Rsequential_12/random_flip_6/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Jsequential_12/random_flip_6/stateless_random_flip_left_right/strided_sliceStridedSliceKsequential_12/random_flip_6/stateless_random_flip_left_right/Shape:output:0Ysequential_12/random_flip_6/stateless_random_flip_left_right/strided_slice/stack:output:0[sequential_12/random_flip_6/stateless_random_flip_left_right/strided_slice/stack_1:output:0[sequential_12/random_flip_6/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskæ
[sequential_12/random_flip_6/stateless_random_flip_left_right/stateless_random_uniform/shapePackSsequential_12/random_flip_6/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:
Ysequential_12/random_flip_6/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
Ysequential_12/random_flip_6/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?è
rsequential_12/random_flip_6/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter2sequential_12/random_flip_6/strided_slice:output:0* 
_output_shapes
::´
rsequential_12/random_flip_6/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
nsequential_12/random_flip_6/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2dsequential_12/random_flip_6/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0xsequential_12/random_flip_6/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0|sequential_12/random_flip_6/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0{sequential_12/random_flip_6/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
Ysequential_12/random_flip_6/stateless_random_flip_left_right/stateless_random_uniform/subSubbsequential_12/random_flip_6/stateless_random_flip_left_right/stateless_random_uniform/max:output:0bsequential_12/random_flip_6/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: æ
Ysequential_12/random_flip_6/stateless_random_flip_left_right/stateless_random_uniform/mulMulwsequential_12/random_flip_6/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0]sequential_12/random_flip_6/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
Usequential_12/random_flip_6/stateless_random_flip_left_right/stateless_random_uniformAddV2]sequential_12/random_flip_6/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0bsequential_12/random_flip_6/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Lsequential_12/random_flip_6/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
Lsequential_12/random_flip_6/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Lsequential_12/random_flip_6/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Ú
Jsequential_12/random_flip_6/stateless_random_flip_left_right/Reshape/shapePackSsequential_12/random_flip_6/stateless_random_flip_left_right/strided_slice:output:0Usequential_12/random_flip_6/stateless_random_flip_left_right/Reshape/shape/1:output:0Usequential_12/random_flip_6/stateless_random_flip_left_right/Reshape/shape/2:output:0Usequential_12/random_flip_6/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:¹
Dsequential_12/random_flip_6/stateless_random_flip_left_right/ReshapeReshapeYsequential_12/random_flip_6/stateless_random_flip_left_right/stateless_random_uniform:z:0Ssequential_12/random_flip_6/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
Bsequential_12/random_flip_6/stateless_random_flip_left_right/RoundRoundMsequential_12/random_flip_6/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ksequential_12/random_flip_6/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:¿
Fsequential_12/random_flip_6/stateless_random_flip_left_right/ReverseV2	ReverseV2Xsequential_12/random_flip_6/stateless_random_flip_left_right/control_dependency:output:0Tsequential_12/random_flip_6/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
@sequential_12/random_flip_6/stateless_random_flip_left_right/mulMulFsequential_12/random_flip_6/stateless_random_flip_left_right/Round:y:0Osequential_12/random_flip_6/stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
Bsequential_12/random_flip_6/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
@sequential_12/random_flip_6/stateless_random_flip_left_right/subSubKsequential_12/random_flip_6/stateless_random_flip_left_right/sub/x:output:0Fsequential_12/random_flip_6/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
Bsequential_12/random_flip_6/stateless_random_flip_left_right/mul_1MulDsequential_12/random_flip_6/stateless_random_flip_left_right/sub:z:0Xsequential_12/random_flip_6/stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
@sequential_12/random_flip_6/stateless_random_flip_left_right/addAddV2Dsequential_12/random_flip_6/stateless_random_flip_left_right/mul:z:0Fsequential_12/random_flip_6/stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
%sequential_12/random_rotation_6/ShapeShapeDsequential_12/random_flip_6/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
:}
3sequential_12/random_rotation_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5sequential_12/random_rotation_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5sequential_12/random_rotation_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
-sequential_12/random_rotation_6/strided_sliceStridedSlice.sequential_12/random_rotation_6/Shape:output:0<sequential_12/random_rotation_6/strided_slice/stack:output:0>sequential_12/random_rotation_6/strided_slice/stack_1:output:0>sequential_12/random_rotation_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5sequential_12/random_rotation_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
7sequential_12/random_rotation_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
7sequential_12/random_rotation_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
/sequential_12/random_rotation_6/strided_slice_1StridedSlice.sequential_12/random_rotation_6/Shape:output:0>sequential_12/random_rotation_6/strided_slice_1/stack:output:0@sequential_12/random_rotation_6/strided_slice_1/stack_1:output:0@sequential_12/random_rotation_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
$sequential_12/random_rotation_6/CastCast8sequential_12/random_rotation_6/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 
5sequential_12/random_rotation_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
7sequential_12/random_rotation_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
7sequential_12/random_rotation_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
/sequential_12/random_rotation_6/strided_slice_2StridedSlice.sequential_12/random_rotation_6/Shape:output:0>sequential_12/random_rotation_6/strided_slice_2/stack:output:0@sequential_12/random_rotation_6/strided_slice_2/stack_1:output:0@sequential_12/random_rotation_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
&sequential_12/random_rotation_6/Cast_1Cast8sequential_12/random_rotation_6/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: ¤
6sequential_12/random_rotation_6/stateful_uniform/shapePack6sequential_12/random_rotation_6/strided_slice:output:0*
N*
T0*
_output_shapes
:y
4sequential_12/random_rotation_6/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ¿y
4sequential_12/random_rotation_6/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ?
6sequential_12/random_rotation_6/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: à
5sequential_12/random_rotation_6/stateful_uniform/ProdProd?sequential_12/random_rotation_6/stateful_uniform/shape:output:0?sequential_12/random_rotation_6/stateful_uniform/Const:output:0*
T0*
_output_shapes
: y
7sequential_12/random_rotation_6/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :¯
7sequential_12/random_rotation_6/stateful_uniform/Cast_1Cast>sequential_12/random_rotation_6/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ¶
?sequential_12/random_rotation_6/stateful_uniform/RngReadAndSkipRngReadAndSkipHsequential_12_random_rotation_6_stateful_uniform_rngreadandskip_resource@sequential_12/random_rotation_6/stateful_uniform/Cast/x:output:0;sequential_12/random_rotation_6/stateful_uniform/Cast_1:y:0*
_output_shapes
:
Dsequential_12/random_rotation_6/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Fsequential_12/random_rotation_6/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Fsequential_12/random_rotation_6/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ì
>sequential_12/random_rotation_6/stateful_uniform/strided_sliceStridedSliceGsequential_12/random_rotation_6/stateful_uniform/RngReadAndSkip:value:0Msequential_12/random_rotation_6/stateful_uniform/strided_slice/stack:output:0Osequential_12/random_rotation_6/stateful_uniform/strided_slice/stack_1:output:0Osequential_12/random_rotation_6/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask½
8sequential_12/random_rotation_6/stateful_uniform/BitcastBitcastGsequential_12/random_rotation_6/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Fsequential_12/random_rotation_6/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Hsequential_12/random_rotation_6/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Hsequential_12/random_rotation_6/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Â
@sequential_12/random_rotation_6/stateful_uniform/strided_slice_1StridedSliceGsequential_12/random_rotation_6/stateful_uniform/RngReadAndSkip:value:0Osequential_12/random_rotation_6/stateful_uniform/strided_slice_1/stack:output:0Qsequential_12/random_rotation_6/stateful_uniform/strided_slice_1/stack_1:output:0Qsequential_12/random_rotation_6/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:Á
:sequential_12/random_rotation_6/stateful_uniform/Bitcast_1BitcastIsequential_12/random_rotation_6/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0
Msequential_12/random_rotation_6/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :«
Isequential_12/random_rotation_6/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2?sequential_12/random_rotation_6/stateful_uniform/shape:output:0Csequential_12/random_rotation_6/stateful_uniform/Bitcast_1:output:0Asequential_12/random_rotation_6/stateful_uniform/Bitcast:output:0Vsequential_12/random_rotation_6/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
4sequential_12/random_rotation_6/stateful_uniform/subSub=sequential_12/random_rotation_6/stateful_uniform/max:output:0=sequential_12/random_rotation_6/stateful_uniform/min:output:0*
T0*
_output_shapes
: ÷
4sequential_12/random_rotation_6/stateful_uniform/mulMulRsequential_12/random_rotation_6/stateful_uniform/StatelessRandomUniformV2:output:08sequential_12/random_rotation_6/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
0sequential_12/random_rotation_6/stateful_uniformAddV28sequential_12/random_rotation_6/stateful_uniform/mul:z:0=sequential_12/random_rotation_6/stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
5sequential_12/random_rotation_6/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ç
3sequential_12/random_rotation_6/rotation_matrix/subSub*sequential_12/random_rotation_6/Cast_1:y:0>sequential_12/random_rotation_6/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 
3sequential_12/random_rotation_6/rotation_matrix/CosCos4sequential_12/random_rotation_6/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
7sequential_12/random_rotation_6/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ë
5sequential_12/random_rotation_6/rotation_matrix/sub_1Sub*sequential_12/random_rotation_6/Cast_1:y:0@sequential_12/random_rotation_6/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: Ü
3sequential_12/random_rotation_6/rotation_matrix/mulMul7sequential_12/random_rotation_6/rotation_matrix/Cos:y:09sequential_12/random_rotation_6/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3sequential_12/random_rotation_6/rotation_matrix/SinSin4sequential_12/random_rotation_6/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
7sequential_12/random_rotation_6/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?É
5sequential_12/random_rotation_6/rotation_matrix/sub_2Sub(sequential_12/random_rotation_6/Cast:y:0@sequential_12/random_rotation_6/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: Þ
5sequential_12/random_rotation_6/rotation_matrix/mul_1Mul7sequential_12/random_rotation_6/rotation_matrix/Sin:y:09sequential_12/random_rotation_6/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ
5sequential_12/random_rotation_6/rotation_matrix/sub_3Sub7sequential_12/random_rotation_6/rotation_matrix/mul:z:09sequential_12/random_rotation_6/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ
5sequential_12/random_rotation_6/rotation_matrix/sub_4Sub7sequential_12/random_rotation_6/rotation_matrix/sub:z:09sequential_12/random_rotation_6/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
9sequential_12/random_rotation_6/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @ï
7sequential_12/random_rotation_6/rotation_matrix/truedivRealDiv9sequential_12/random_rotation_6/rotation_matrix/sub_4:z:0Bsequential_12/random_rotation_6/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
7sequential_12/random_rotation_6/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?É
5sequential_12/random_rotation_6/rotation_matrix/sub_5Sub(sequential_12/random_rotation_6/Cast:y:0@sequential_12/random_rotation_6/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
:  
5sequential_12/random_rotation_6/rotation_matrix/Sin_1Sin4sequential_12/random_rotation_6/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
7sequential_12/random_rotation_6/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ë
5sequential_12/random_rotation_6/rotation_matrix/sub_6Sub*sequential_12/random_rotation_6/Cast_1:y:0@sequential_12/random_rotation_6/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: à
5sequential_12/random_rotation_6/rotation_matrix/mul_2Mul9sequential_12/random_rotation_6/rotation_matrix/Sin_1:y:09sequential_12/random_rotation_6/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
5sequential_12/random_rotation_6/rotation_matrix/Cos_1Cos4sequential_12/random_rotation_6/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
7sequential_12/random_rotation_6/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?É
5sequential_12/random_rotation_6/rotation_matrix/sub_7Sub(sequential_12/random_rotation_6/Cast:y:0@sequential_12/random_rotation_6/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: à
5sequential_12/random_rotation_6/rotation_matrix/mul_3Mul9sequential_12/random_rotation_6/rotation_matrix/Cos_1:y:09sequential_12/random_rotation_6/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
3sequential_12/random_rotation_6/rotation_matrix/addAddV29sequential_12/random_rotation_6/rotation_matrix/mul_2:z:09sequential_12/random_rotation_6/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ
5sequential_12/random_rotation_6/rotation_matrix/sub_8Sub9sequential_12/random_rotation_6/rotation_matrix/sub_5:z:07sequential_12/random_rotation_6/rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
;sequential_12/random_rotation_6/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @ó
9sequential_12/random_rotation_6/rotation_matrix/truediv_1RealDiv9sequential_12/random_rotation_6/rotation_matrix/sub_8:z:0Dsequential_12/random_rotation_6/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5sequential_12/random_rotation_6/rotation_matrix/ShapeShape4sequential_12/random_rotation_6/stateful_uniform:z:0*
T0*
_output_shapes
:
Csequential_12/random_rotation_6/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Esequential_12/random_rotation_6/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Esequential_12/random_rotation_6/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Á
=sequential_12/random_rotation_6/rotation_matrix/strided_sliceStridedSlice>sequential_12/random_rotation_6/rotation_matrix/Shape:output:0Lsequential_12/random_rotation_6/rotation_matrix/strided_slice/stack:output:0Nsequential_12/random_rotation_6/rotation_matrix/strided_slice/stack_1:output:0Nsequential_12/random_rotation_6/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask 
5sequential_12/random_rotation_6/rotation_matrix/Cos_2Cos4sequential_12/random_rotation_6/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Esequential_12/random_rotation_6/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Gsequential_12/random_rotation_6/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Gsequential_12/random_rotation_6/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
?sequential_12/random_rotation_6/rotation_matrix/strided_slice_1StridedSlice9sequential_12/random_rotation_6/rotation_matrix/Cos_2:y:0Nsequential_12/random_rotation_6/rotation_matrix/strided_slice_1/stack:output:0Psequential_12/random_rotation_6/rotation_matrix/strided_slice_1/stack_1:output:0Psequential_12/random_rotation_6/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask 
5sequential_12/random_rotation_6/rotation_matrix/Sin_2Sin4sequential_12/random_rotation_6/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Esequential_12/random_rotation_6/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Gsequential_12/random_rotation_6/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Gsequential_12/random_rotation_6/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
?sequential_12/random_rotation_6/rotation_matrix/strided_slice_2StridedSlice9sequential_12/random_rotation_6/rotation_matrix/Sin_2:y:0Nsequential_12/random_rotation_6/rotation_matrix/strided_slice_2/stack:output:0Psequential_12/random_rotation_6/rotation_matrix/strided_slice_2/stack_1:output:0Psequential_12/random_rotation_6/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask¶
3sequential_12/random_rotation_6/rotation_matrix/NegNegHsequential_12/random_rotation_6/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Esequential_12/random_rotation_6/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Gsequential_12/random_rotation_6/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Gsequential_12/random_rotation_6/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ö
?sequential_12/random_rotation_6/rotation_matrix/strided_slice_3StridedSlice;sequential_12/random_rotation_6/rotation_matrix/truediv:z:0Nsequential_12/random_rotation_6/rotation_matrix/strided_slice_3/stack:output:0Psequential_12/random_rotation_6/rotation_matrix/strided_slice_3/stack_1:output:0Psequential_12/random_rotation_6/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask 
5sequential_12/random_rotation_6/rotation_matrix/Sin_3Sin4sequential_12/random_rotation_6/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Esequential_12/random_rotation_6/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Gsequential_12/random_rotation_6/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Gsequential_12/random_rotation_6/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
?sequential_12/random_rotation_6/rotation_matrix/strided_slice_4StridedSlice9sequential_12/random_rotation_6/rotation_matrix/Sin_3:y:0Nsequential_12/random_rotation_6/rotation_matrix/strided_slice_4/stack:output:0Psequential_12/random_rotation_6/rotation_matrix/strided_slice_4/stack_1:output:0Psequential_12/random_rotation_6/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask 
5sequential_12/random_rotation_6/rotation_matrix/Cos_3Cos4sequential_12/random_rotation_6/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Esequential_12/random_rotation_6/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Gsequential_12/random_rotation_6/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Gsequential_12/random_rotation_6/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
?sequential_12/random_rotation_6/rotation_matrix/strided_slice_5StridedSlice9sequential_12/random_rotation_6/rotation_matrix/Cos_3:y:0Nsequential_12/random_rotation_6/rotation_matrix/strided_slice_5/stack:output:0Psequential_12/random_rotation_6/rotation_matrix/strided_slice_5/stack_1:output:0Psequential_12/random_rotation_6/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
Esequential_12/random_rotation_6/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Gsequential_12/random_rotation_6/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Gsequential_12/random_rotation_6/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ø
?sequential_12/random_rotation_6/rotation_matrix/strided_slice_6StridedSlice=sequential_12/random_rotation_6/rotation_matrix/truediv_1:z:0Nsequential_12/random_rotation_6/rotation_matrix/strided_slice_6/stack:output:0Psequential_12/random_rotation_6/rotation_matrix/strided_slice_6/stack_1:output:0Psequential_12/random_rotation_6/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
>sequential_12/random_rotation_6/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
<sequential_12/random_rotation_6/rotation_matrix/zeros/packedPackFsequential_12/random_rotation_6/rotation_matrix/strided_slice:output:0Gsequential_12/random_rotation_6/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:
;sequential_12/random_rotation_6/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ü
5sequential_12/random_rotation_6/rotation_matrix/zerosFillEsequential_12/random_rotation_6/rotation_matrix/zeros/packed:output:0Dsequential_12/random_rotation_6/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
;sequential_12/random_rotation_6/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :®
6sequential_12/random_rotation_6/rotation_matrix/concatConcatV2Hsequential_12/random_rotation_6/rotation_matrix/strided_slice_1:output:07sequential_12/random_rotation_6/rotation_matrix/Neg:y:0Hsequential_12/random_rotation_6/rotation_matrix/strided_slice_3:output:0Hsequential_12/random_rotation_6/rotation_matrix/strided_slice_4:output:0Hsequential_12/random_rotation_6/rotation_matrix/strided_slice_5:output:0Hsequential_12/random_rotation_6/rotation_matrix/strided_slice_6:output:0>sequential_12/random_rotation_6/rotation_matrix/zeros:output:0Dsequential_12/random_rotation_6/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
/sequential_12/random_rotation_6/transform/ShapeShapeDsequential_12/random_flip_6/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
:
=sequential_12/random_rotation_6/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?sequential_12/random_rotation_6/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?sequential_12/random_rotation_6/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
7sequential_12/random_rotation_6/transform/strided_sliceStridedSlice8sequential_12/random_rotation_6/transform/Shape:output:0Fsequential_12/random_rotation_6/transform/strided_slice/stack:output:0Hsequential_12/random_rotation_6/transform/strided_slice/stack_1:output:0Hsequential_12/random_rotation_6/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:y
4sequential_12/random_rotation_6/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ß
Dsequential_12/random_rotation_6/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Dsequential_12/random_flip_6/stateless_random_flip_left_right/add:z:0?sequential_12/random_rotation_6/rotation_matrix/concat:output:0@sequential_12/random_rotation_6/transform/strided_slice:output:0=sequential_12/random_rotation_6/transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEARª
!sequential_12/random_zoom_6/ShapeShapeYsequential_12/random_rotation_6/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:y
/sequential_12/random_zoom_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_12/random_zoom_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_12/random_zoom_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)sequential_12/random_zoom_6/strided_sliceStridedSlice*sequential_12/random_zoom_6/Shape:output:08sequential_12/random_zoom_6/strided_slice/stack:output:0:sequential_12/random_zoom_6/strided_slice/stack_1:output:0:sequential_12/random_zoom_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
1sequential_12/random_zoom_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
3sequential_12/random_zoom_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ}
3sequential_12/random_zoom_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+sequential_12/random_zoom_6/strided_slice_1StridedSlice*sequential_12/random_zoom_6/Shape:output:0:sequential_12/random_zoom_6/strided_slice_1/stack:output:0<sequential_12/random_zoom_6/strided_slice_1/stack_1:output:0<sequential_12/random_zoom_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
 sequential_12/random_zoom_6/CastCast4sequential_12/random_zoom_6/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 
1sequential_12/random_zoom_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
3sequential_12/random_zoom_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ}
3sequential_12/random_zoom_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+sequential_12/random_zoom_6/strided_slice_2StridedSlice*sequential_12/random_zoom_6/Shape:output:0:sequential_12/random_zoom_6/strided_slice_2/stack:output:0<sequential_12/random_zoom_6/strided_slice_2/stack_1:output:0<sequential_12/random_zoom_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
"sequential_12/random_zoom_6/Cast_1Cast4sequential_12/random_zoom_6/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: v
4sequential_12/random_zoom_6/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Û
2sequential_12/random_zoom_6/stateful_uniform/shapePack2sequential_12/random_zoom_6/strided_slice:output:0=sequential_12/random_zoom_6/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:u
0sequential_12/random_zoom_6/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?u
0sequential_12/random_zoom_6/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?|
2sequential_12/random_zoom_6/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ô
1sequential_12/random_zoom_6/stateful_uniform/ProdProd;sequential_12/random_zoom_6/stateful_uniform/shape:output:0;sequential_12/random_zoom_6/stateful_uniform/Const:output:0*
T0*
_output_shapes
: u
3sequential_12/random_zoom_6/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :§
3sequential_12/random_zoom_6/stateful_uniform/Cast_1Cast:sequential_12/random_zoom_6/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ¦
;sequential_12/random_zoom_6/stateful_uniform/RngReadAndSkipRngReadAndSkipDsequential_12_random_zoom_6_stateful_uniform_rngreadandskip_resource<sequential_12/random_zoom_6/stateful_uniform/Cast/x:output:07sequential_12/random_zoom_6/stateful_uniform/Cast_1:y:0*
_output_shapes
:
@sequential_12/random_zoom_6/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Bsequential_12/random_zoom_6/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Bsequential_12/random_zoom_6/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
:sequential_12/random_zoom_6/stateful_uniform/strided_sliceStridedSliceCsequential_12/random_zoom_6/stateful_uniform/RngReadAndSkip:value:0Isequential_12/random_zoom_6/stateful_uniform/strided_slice/stack:output:0Ksequential_12/random_zoom_6/stateful_uniform/strided_slice/stack_1:output:0Ksequential_12/random_zoom_6/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskµ
4sequential_12/random_zoom_6/stateful_uniform/BitcastBitcastCsequential_12/random_zoom_6/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Bsequential_12/random_zoom_6/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Dsequential_12/random_zoom_6/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Dsequential_12/random_zoom_6/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
<sequential_12/random_zoom_6/stateful_uniform/strided_slice_1StridedSliceCsequential_12/random_zoom_6/stateful_uniform/RngReadAndSkip:value:0Ksequential_12/random_zoom_6/stateful_uniform/strided_slice_1/stack:output:0Msequential_12/random_zoom_6/stateful_uniform/strided_slice_1/stack_1:output:0Msequential_12/random_zoom_6/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:¹
6sequential_12/random_zoom_6/stateful_uniform/Bitcast_1BitcastEsequential_12/random_zoom_6/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0
Isequential_12/random_zoom_6/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
Esequential_12/random_zoom_6/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2;sequential_12/random_zoom_6/stateful_uniform/shape:output:0?sequential_12/random_zoom_6/stateful_uniform/Bitcast_1:output:0=sequential_12/random_zoom_6/stateful_uniform/Bitcast:output:0Rsequential_12/random_zoom_6/stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
0sequential_12/random_zoom_6/stateful_uniform/subSub9sequential_12/random_zoom_6/stateful_uniform/max:output:09sequential_12/random_zoom_6/stateful_uniform/min:output:0*
T0*
_output_shapes
: ï
0sequential_12/random_zoom_6/stateful_uniform/mulMulNsequential_12/random_zoom_6/stateful_uniform/StatelessRandomUniformV2:output:04sequential_12/random_zoom_6/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
,sequential_12/random_zoom_6/stateful_uniformAddV24sequential_12/random_zoom_6/stateful_uniform/mul:z:09sequential_12/random_zoom_6/stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
'sequential_12/random_zoom_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ÿ
"sequential_12/random_zoom_6/concatConcatV20sequential_12/random_zoom_6/stateful_uniform:z:00sequential_12/random_zoom_6/stateful_uniform:z:00sequential_12/random_zoom_6/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
-sequential_12/random_zoom_6/zoom_matrix/ShapeShape+sequential_12/random_zoom_6/concat:output:0*
T0*
_output_shapes
:
;sequential_12/random_zoom_6/zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=sequential_12/random_zoom_6/zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=sequential_12/random_zoom_6/zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5sequential_12/random_zoom_6/zoom_matrix/strided_sliceStridedSlice6sequential_12/random_zoom_6/zoom_matrix/Shape:output:0Dsequential_12/random_zoom_6/zoom_matrix/strided_slice/stack:output:0Fsequential_12/random_zoom_6/zoom_matrix/strided_slice/stack_1:output:0Fsequential_12/random_zoom_6/zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
-sequential_12/random_zoom_6/zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?³
+sequential_12/random_zoom_6/zoom_matrix/subSub&sequential_12/random_zoom_6/Cast_1:y:06sequential_12/random_zoom_6/zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: v
1sequential_12/random_zoom_6/zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @È
/sequential_12/random_zoom_6/zoom_matrix/truedivRealDiv/sequential_12/random_zoom_6/zoom_matrix/sub:z:0:sequential_12/random_zoom_6/zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 
=sequential_12/random_zoom_6/zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
?sequential_12/random_zoom_6/zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?sequential_12/random_zoom_6/zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Þ
7sequential_12/random_zoom_6/zoom_matrix/strided_slice_1StridedSlice+sequential_12/random_zoom_6/concat:output:0Fsequential_12/random_zoom_6/zoom_matrix/strided_slice_1/stack:output:0Hsequential_12/random_zoom_6/zoom_matrix/strided_slice_1/stack_1:output:0Hsequential_12/random_zoom_6/zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskt
/sequential_12/random_zoom_6/zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?â
-sequential_12/random_zoom_6/zoom_matrix/sub_1Sub8sequential_12/random_zoom_6/zoom_matrix/sub_1/x:output:0@sequential_12/random_zoom_6/zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
+sequential_12/random_zoom_6/zoom_matrix/mulMul3sequential_12/random_zoom_6/zoom_matrix/truediv:z:01sequential_12/random_zoom_6/zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
/sequential_12/random_zoom_6/zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?µ
-sequential_12/random_zoom_6/zoom_matrix/sub_2Sub$sequential_12/random_zoom_6/Cast:y:08sequential_12/random_zoom_6/zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: x
3sequential_12/random_zoom_6/zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Î
1sequential_12/random_zoom_6/zoom_matrix/truediv_1RealDiv1sequential_12/random_zoom_6/zoom_matrix/sub_2:z:0<sequential_12/random_zoom_6/zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 
=sequential_12/random_zoom_6/zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?sequential_12/random_zoom_6/zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?sequential_12/random_zoom_6/zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Þ
7sequential_12/random_zoom_6/zoom_matrix/strided_slice_2StridedSlice+sequential_12/random_zoom_6/concat:output:0Fsequential_12/random_zoom_6/zoom_matrix/strided_slice_2/stack:output:0Hsequential_12/random_zoom_6/zoom_matrix/strided_slice_2/stack_1:output:0Hsequential_12/random_zoom_6/zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskt
/sequential_12/random_zoom_6/zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?â
-sequential_12/random_zoom_6/zoom_matrix/sub_3Sub8sequential_12/random_zoom_6/zoom_matrix/sub_3/x:output:0@sequential_12/random_zoom_6/zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
-sequential_12/random_zoom_6/zoom_matrix/mul_1Mul5sequential_12/random_zoom_6/zoom_matrix/truediv_1:z:01sequential_12/random_zoom_6/zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
=sequential_12/random_zoom_6/zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
?sequential_12/random_zoom_6/zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?sequential_12/random_zoom_6/zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Þ
7sequential_12/random_zoom_6/zoom_matrix/strided_slice_3StridedSlice+sequential_12/random_zoom_6/concat:output:0Fsequential_12/random_zoom_6/zoom_matrix/strided_slice_3/stack:output:0Hsequential_12/random_zoom_6/zoom_matrix/strided_slice_3/stack_1:output:0Hsequential_12/random_zoom_6/zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskx
6sequential_12/random_zoom_6/zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ë
4sequential_12/random_zoom_6/zoom_matrix/zeros/packedPack>sequential_12/random_zoom_6/zoom_matrix/strided_slice:output:0?sequential_12/random_zoom_6/zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:x
3sequential_12/random_zoom_6/zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ä
-sequential_12/random_zoom_6/zoom_matrix/zerosFill=sequential_12/random_zoom_6/zoom_matrix/zeros/packed:output:0<sequential_12/random_zoom_6/zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
8sequential_12/random_zoom_6/zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ï
6sequential_12/random_zoom_6/zoom_matrix/zeros_1/packedPack>sequential_12/random_zoom_6/zoom_matrix/strided_slice:output:0Asequential_12/random_zoom_6/zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:z
5sequential_12/random_zoom_6/zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ê
/sequential_12/random_zoom_6/zoom_matrix/zeros_1Fill?sequential_12/random_zoom_6/zoom_matrix/zeros_1/packed:output:0>sequential_12/random_zoom_6/zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
=sequential_12/random_zoom_6/zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
?sequential_12/random_zoom_6/zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
?sequential_12/random_zoom_6/zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Þ
7sequential_12/random_zoom_6/zoom_matrix/strided_slice_4StridedSlice+sequential_12/random_zoom_6/concat:output:0Fsequential_12/random_zoom_6/zoom_matrix/strided_slice_4/stack:output:0Hsequential_12/random_zoom_6/zoom_matrix/strided_slice_4/stack_1:output:0Hsequential_12/random_zoom_6/zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskz
8sequential_12/random_zoom_6/zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ï
6sequential_12/random_zoom_6/zoom_matrix/zeros_2/packedPack>sequential_12/random_zoom_6/zoom_matrix/strided_slice:output:0Asequential_12/random_zoom_6/zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:z
5sequential_12/random_zoom_6/zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ê
/sequential_12/random_zoom_6/zoom_matrix/zeros_2Fill?sequential_12/random_zoom_6/zoom_matrix/zeros_2/packed:output:0>sequential_12/random_zoom_6/zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
3sequential_12/random_zoom_6/zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ç
.sequential_12/random_zoom_6/zoom_matrix/concatConcatV2@sequential_12/random_zoom_6/zoom_matrix/strided_slice_3:output:06sequential_12/random_zoom_6/zoom_matrix/zeros:output:0/sequential_12/random_zoom_6/zoom_matrix/mul:z:08sequential_12/random_zoom_6/zoom_matrix/zeros_1:output:0@sequential_12/random_zoom_6/zoom_matrix/strided_slice_4:output:01sequential_12/random_zoom_6/zoom_matrix/mul_1:z:08sequential_12/random_zoom_6/zoom_matrix/zeros_2:output:0<sequential_12/random_zoom_6/zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
+sequential_12/random_zoom_6/transform/ShapeShapeYsequential_12/random_rotation_6/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:
9sequential_12/random_zoom_6/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
;sequential_12/random_zoom_6/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;sequential_12/random_zoom_6/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:û
3sequential_12/random_zoom_6/transform/strided_sliceStridedSlice4sequential_12/random_zoom_6/transform/Shape:output:0Bsequential_12/random_zoom_6/transform/strided_slice/stack:output:0Dsequential_12/random_zoom_6/transform/strided_slice/stack_1:output:0Dsequential_12/random_zoom_6/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:u
0sequential_12/random_zoom_6/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    à
@sequential_12/random_zoom_6/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Ysequential_12/random_rotation_6/transform/ImageProjectiveTransformV3:transformed_images:07sequential_12/random_zoom_6/zoom_matrix/concat:output:0<sequential_12/random_zoom_6/transform/strided_slice:output:09sequential_12/random_zoom_6/transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEARW
rescaling_6/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;Y
rescaling_6/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Æ
rescaling_6/mulMulUsequential_12/random_zoom_6/transform/ImageProjectiveTransformV3:transformed_images:0rescaling_6/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
rescaling_6/addAddV2rescaling_6/mul:z:0rescaling_6/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0½
conv2d_22/Conv2DConv2Drescaling_6/add:z:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²²*
paddingVALID*
strides

 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²²n
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²²®
max_pooling2d_22/MaxPoolMaxPoolconv2d_22/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿYY*
ksize
*
paddingVALID*
strides

conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0É
conv2d_23/Conv2DConv2D!max_pooling2d_22/MaxPool:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿWW *
paddingVALID*
strides

 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿWW l
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿWW ®
max_pooling2d_23/MaxPoolMaxPoolconv2d_23/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ++ *
ksize
*
paddingVALID*
strides

conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0É
conv2d_24/Conv2DConv2D!max_pooling2d_23/MaxPool:output:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@*
paddingVALID*
strides

 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@l
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@®
max_pooling2d_24/MaxPoolMaxPoolconv2d_24/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0É
conv2d_25/Conv2DConv2D!max_pooling2d_24/MaxPool:output:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@®
max_pooling2d_25/MaxPoolMaxPoolconv2d_25/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*
ksize
*
paddingVALID*
strides
`
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  
flatten_6/ReshapeReshape!max_pooling2d_25/MaxPool:output:0flatten_6/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ(
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
À(*
dtype0
dense_12/MatMulMatMulflatten_6/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_13/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
NoOpNoOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOpE^sequential_12/random_flip_6/stateful_uniform_full_int/RngReadAndSkip@^sequential_12/random_rotation_6/stateful_uniform/RngReadAndSkip<^sequential_12/random_zoom_6/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ´´: : : : : : : : : : : : : : : 2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp2
Dsequential_12/random_flip_6/stateful_uniform_full_int/RngReadAndSkipDsequential_12/random_flip_6/stateful_uniform_full_int/RngReadAndSkip2
?sequential_12/random_rotation_6/stateful_uniform/RngReadAndSkip?sequential_12/random_rotation_6/stateful_uniform/RngReadAndSkip2z
;sequential_12/random_zoom_6/stateful_uniform/RngReadAndSkip;sequential_12/random_zoom_6/stateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs

ý
D__inference_conv2d_23_layer_call_and_return_conditional_losses_45270

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿWW *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿWW X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿWW i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿWW w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿYY: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿYY
 
_user_specified_nameinputs
Ò5
ê
H__inference_sequential_13_layer_call_and_return_conditional_losses_45642
input_7)
conv2d_22_45606:
conv2d_22_45608:)
conv2d_23_45612: 
conv2d_23_45614: )
conv2d_24_45618: @
conv2d_24_45620:@)
conv2d_25_45624:@@
conv2d_25_45626:@"
dense_12_45631:
À(
dense_12_45633:	!
dense_13_45636:	
dense_13_45638:
identity¢!conv2d_22/StatefulPartitionedCall¢!conv2d_23/StatefulPartitionedCall¢!conv2d_24/StatefulPartitionedCall¢!conv2d_25/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCallÏ
sequential_12/PartitionedCallPartitionedCallinput_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_12_layer_call_and_return_conditional_losses_44795ê
rescaling_6/PartitionedCallPartitionedCall&sequential_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_rescaling_6_layer_call_and_return_conditional_losses_45239
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall$rescaling_6/PartitionedCall:output:0conv2d_22_45606conv2d_22_45608*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²²*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_45252ö
 max_pooling2d_22/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿYY* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_45184
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_22/PartitionedCall:output:0conv2d_23_45612conv2d_23_45614*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿWW *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_45270ö
 max_pooling2d_23/PartitionedCallPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ++ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_45196
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_23/PartitionedCall:output:0conv2d_24_45618conv2d_24_45620*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_45288ö
 max_pooling2d_24/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_45208
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_24/PartitionedCall:output:0conv2d_25_45624conv2d_25_45626*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_45306ö
 max_pooling2d_25/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_45220à
flatten_6/PartitionedCallPartitionedCall)max_pooling2d_25/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_45319
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_12_45631dense_12_45633*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_45332
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_45636dense_13_45638*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_45349x
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ´´: : : : : : : : : : : : 2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
!
_user_specified_name	input_7
ø
b
F__inference_rescaling_6_layer_call_and_return_conditional_losses_45239

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    _
mulMulinputsCast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´d
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´Y
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´´:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_46501

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í

)__inference_conv2d_24_layer_call_fn_46540

inputs!
unknown: @
	unknown_0:@
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_45288w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ++ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ++ 
 
_user_specified_nameinputs

d
H__inference_sequential_12_layer_call_and_return_conditional_losses_46188

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´´:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
£

õ
C__inference_dense_13_layer_call_and_return_conditional_losses_46642

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦

-__inference_sequential_13_layer_call_fn_45759

inputs
unknown:	
	unknown_0:	
	unknown_1:	#
	unknown_2:
	unknown_3:#
	unknown_4: 
	unknown_5: #
	unknown_6: @
	unknown_7:@#
	unknown_8:@@
	unknown_9:@

unknown_10:
À(

unknown_11:	

unknown_12:	

unknown_13:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_13_layer_call_and_return_conditional_losses_45533o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ´´: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
Ï5
é
H__inference_sequential_13_layer_call_and_return_conditional_losses_45356

inputs)
conv2d_22_45253:
conv2d_22_45255:)
conv2d_23_45271: 
conv2d_23_45273: )
conv2d_24_45289: @
conv2d_24_45291:@)
conv2d_25_45307:@@
conv2d_25_45309:@"
dense_12_45333:
À(
dense_12_45335:	!
dense_13_45350:	
dense_13_45352:
identity¢!conv2d_22/StatefulPartitionedCall¢!conv2d_23/StatefulPartitionedCall¢!conv2d_24/StatefulPartitionedCall¢!conv2d_25/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCallÎ
sequential_12/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_12_layer_call_and_return_conditional_losses_44795ê
rescaling_6/PartitionedCallPartitionedCall&sequential_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_rescaling_6_layer_call_and_return_conditional_losses_45239
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall$rescaling_6/PartitionedCall:output:0conv2d_22_45253conv2d_22_45255*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²²*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_45252ö
 max_pooling2d_22/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿYY* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_45184
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_22/PartitionedCall:output:0conv2d_23_45271conv2d_23_45273*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿWW *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_45270ö
 max_pooling2d_23/PartitionedCallPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ++ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_45196
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_23/PartitionedCall:output:0conv2d_24_45289conv2d_24_45291*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_45288ö
 max_pooling2d_24/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_45208
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_24/PartitionedCall:output:0conv2d_25_45307conv2d_25_45309*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_45306ö
 max_pooling2d_25/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_45220à
flatten_6/PartitionedCallPartitionedCall)max_pooling2d_25/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_45319
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_12_45333dense_12_45335*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_45332
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_45350dense_13_45352*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_45349x
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ´´: : : : : : : : : : : : 2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
÷
V
-__inference_sequential_12_layer_call_fn_44798
random_flip_6_input
identityÍ
PartitionedCallPartitionedCallrandom_flip_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_12_layer_call_and_return_conditional_losses_44795j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´´:f b
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
-
_user_specified_namerandom_flip_6_input

¯
-__inference_sequential_12_layer_call_fn_46184

inputs
unknown:	
	unknown_0:	
	unknown_1:	
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_12_layer_call_and_return_conditional_losses_45135y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ´´: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
²
E
)__inference_flatten_6_layer_call_fn_46596

inputs
identity³
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_45319a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ		@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@
 
_user_specified_nameinputs

ý
D__inference_conv2d_22_layer_call_and_return_conditional_losses_46491

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²²*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²²Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²²k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²²w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ´´: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_46561

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

d
H__inference_random_zoom_6_layer_call_and_return_conditional_losses_44792

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´´:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
ÉC
ß	
H__inference_sequential_13_layer_call_and_return_conditional_losses_45815

inputsB
(conv2d_22_conv2d_readvariableop_resource:7
)conv2d_22_biasadd_readvariableop_resource:B
(conv2d_23_conv2d_readvariableop_resource: 7
)conv2d_23_biasadd_readvariableop_resource: B
(conv2d_24_conv2d_readvariableop_resource: @7
)conv2d_24_biasadd_readvariableop_resource:@B
(conv2d_25_conv2d_readvariableop_resource:@@7
)conv2d_25_biasadd_readvariableop_resource:@;
'dense_12_matmul_readvariableop_resource:
À(7
(dense_12_biasadd_readvariableop_resource:	:
'dense_13_matmul_readvariableop_resource:	6
(dense_13_biasadd_readvariableop_resource:
identity¢ conv2d_22/BiasAdd/ReadVariableOp¢conv2d_22/Conv2D/ReadVariableOp¢ conv2d_23/BiasAdd/ReadVariableOp¢conv2d_23/Conv2D/ReadVariableOp¢ conv2d_24/BiasAdd/ReadVariableOp¢conv2d_24/Conv2D/ReadVariableOp¢ conv2d_25/BiasAdd/ReadVariableOp¢conv2d_25/Conv2D/ReadVariableOp¢dense_12/BiasAdd/ReadVariableOp¢dense_12/MatMul/ReadVariableOp¢dense_13/BiasAdd/ReadVariableOp¢dense_13/MatMul/ReadVariableOpW
rescaling_6/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;Y
rescaling_6/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    w
rescaling_6/mulMulinputsrescaling_6/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
rescaling_6/addAddV2rescaling_6/mul:z:0rescaling_6/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0½
conv2d_22/Conv2DConv2Drescaling_6/add:z:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²²*
paddingVALID*
strides

 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²²n
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²²®
max_pooling2d_22/MaxPoolMaxPoolconv2d_22/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿYY*
ksize
*
paddingVALID*
strides

conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0É
conv2d_23/Conv2DConv2D!max_pooling2d_22/MaxPool:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿWW *
paddingVALID*
strides

 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿWW l
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿWW ®
max_pooling2d_23/MaxPoolMaxPoolconv2d_23/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ++ *
ksize
*
paddingVALID*
strides

conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0É
conv2d_24/Conv2DConv2D!max_pooling2d_23/MaxPool:output:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@*
paddingVALID*
strides

 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@l
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@®
max_pooling2d_24/MaxPoolMaxPoolconv2d_24/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0É
conv2d_25/Conv2DConv2D!max_pooling2d_24/MaxPool:output:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@®
max_pooling2d_25/MaxPoolMaxPoolconv2d_25/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*
ksize
*
paddingVALID*
strides
`
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  
flatten_6/ReshapeReshape!max_pooling2d_25/MaxPool:output:0flatten_6/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ(
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
À(*
dtype0
dense_12/MatMulMatMulflatten_6/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_13/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
NoOpNoOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ´´: : : : : : : : : : : : 2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
£

õ
C__inference_dense_13_layer_call_and_return_conditional_losses_45349

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ý
D__inference_conv2d_24_layer_call_and_return_conditional_losses_46551

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ++ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ++ 
 
_user_specified_nameinputs
ü

Ä
#__inference_signature_wrapper_46168
input_7!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
À(
	unknown_8:	
	unknown_9:	

unknown_10:
identity¢StatefulPartitionedCall¼
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_44769o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ´´: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
!
_user_specified_name	input_7
¦

÷
C__inference_dense_12_layer_call_and_return_conditional_losses_45332

inputs2
matmul_readvariableop_resource:
À(.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
À(*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ(
 
_user_specified_nameinputs
»
¼
-__inference_sequential_12_layer_call_fn_45155
random_flip_6_input
unknown:	
	unknown_0:	
	unknown_1:	
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallrandom_flip_6_inputunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_12_layer_call_and_return_conditional_losses_45135y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ´´: : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
-
_user_specified_namerandom_flip_6_input
Å
}
-__inference_random_flip_6_layer_call_fn_46654

inputs
unknown:	
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_random_flip_6_layer_call_and_return_conditional_losses_45110y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ´´: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
ÝM
Ó
H__inference_random_flip_6_layer_call_and_return_conditional_losses_46716

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity¢(stateful_uniform_full_int/RngReadAndSkipi
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: b
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Ú
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:w
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0y
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ï
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0_
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	T

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R x
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask
3stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
&stateless_random_flip_left_right/ShapeShape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:~
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask®
?stateless_random_flip_left_right/stateless_random_uniform/shapePack7stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :þ
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0_stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿõ
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿû
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Î
.stateless_random_flip_left_right/Reshape/shapePack7stateless_random_flip_left_right/strided_slice:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:å
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:ë
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´È
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´k
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Â
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´¿
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
IdentityIdentity(stateless_random_flip_left_right/add:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´q
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ´´: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_45196

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»8
û
H__inference_sequential_13_layer_call_and_return_conditional_losses_45689
input_7!
sequential_12_45645:	!
sequential_12_45647:	!
sequential_12_45649:	)
conv2d_22_45653:
conv2d_22_45655:)
conv2d_23_45659: 
conv2d_23_45661: )
conv2d_24_45665: @
conv2d_24_45667:@)
conv2d_25_45671:@@
conv2d_25_45673:@"
dense_12_45678:
À(
dense_12_45680:	!
dense_13_45683:	
dense_13_45685:
identity¢!conv2d_22/StatefulPartitionedCall¢!conv2d_23/StatefulPartitionedCall¢!conv2d_24/StatefulPartitionedCall¢!conv2d_25/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢%sequential_12/StatefulPartitionedCall¡
%sequential_12/StatefulPartitionedCallStatefulPartitionedCallinput_7sequential_12_45645sequential_12_45647sequential_12_45649*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_12_layer_call_and_return_conditional_losses_45135ò
rescaling_6/PartitionedCallPartitionedCall.sequential_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_rescaling_6_layer_call_and_return_conditional_losses_45239
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall$rescaling_6/PartitionedCall:output:0conv2d_22_45653conv2d_22_45655*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²²*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_45252ö
 max_pooling2d_22/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿYY* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_45184
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_22/PartitionedCall:output:0conv2d_23_45659conv2d_23_45661*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿWW *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_45270ö
 max_pooling2d_23/PartitionedCallPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ++ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_45196
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_23/PartitionedCall:output:0conv2d_24_45665conv2d_24_45667*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_45288ö
 max_pooling2d_24/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_45208
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_24/PartitionedCall:output:0conv2d_25_45671conv2d_25_45673*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_45306ö
 max_pooling2d_25/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_45220à
flatten_6/PartitionedCallPartitionedCall)max_pooling2d_25/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_45319
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_12_45678dense_12_45680*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_45332
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_45683dense_13_45685*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_45349x
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
NoOpNoOp"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall&^sequential_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ´´: : : : : : : : : : : : : : : 2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2N
%sequential_12/StatefulPartitionedCall%sequential_12/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
!
_user_specified_name	input_7
n
Á
H__inference_random_zoom_6_layer_call_and_return_conditional_losses_46968

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity¢stateful_uniform/RngReadAndSkip;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿj
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿa
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿj
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: Z
stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
stateful_uniform/shapePackstrided_slice:output:0!stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?Y
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ¶
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¢
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2stateful_uniform:z:0stateful_uniform:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
zoom_matrix/ShapeShapeconcat:output:0*
T0*
_output_shapes
:i
zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
zoom_matrix/strided_sliceStridedSlicezoom_matrix/Shape:output:0(zoom_matrix/strided_slice/stack:output:0*zoom_matrix/strided_slice/stack_1:output:0*zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
zoom_matrix/subSub
Cast_1:y:0zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: Z
zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @t
zoom_matrix/truedivRealDivzoom_matrix/sub:z:0zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: v
!zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            x
#zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_1StridedSliceconcat:output:0*zoom_matrix/strided_slice_1/stack:output:0,zoom_matrix/strided_slice_1/stack_1:output:0,zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskX
zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
zoom_matrix/sub_1Subzoom_matrix/sub_1/x:output:0$zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
zoom_matrix/mulMulzoom_matrix/truediv:z:0zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
zoom_matrix/sub_2SubCast:y:0zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: \
zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @z
zoom_matrix/truediv_1RealDivzoom_matrix/sub_2:z:0 zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: v
!zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_2StridedSliceconcat:output:0*zoom_matrix/strided_slice_2/stack:output:0,zoom_matrix/strided_slice_2/stack_1:output:0,zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskX
zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
zoom_matrix/sub_3Subzoom_matrix/sub_3/x:output:0$zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
zoom_matrix/mul_1Mulzoom_matrix/truediv_1:z:0zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
!zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            x
#zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_3StridedSliceconcat:output:0*zoom_matrix/strided_slice_3/stack:output:0,zoom_matrix/strided_slice_3/stack_1:output:0,zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask\
zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
zoom_matrix/zeros/packedPack"zoom_matrix/strided_slice:output:0#zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:\
zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
zoom_matrix/zerosFill!zoom_matrix/zeros/packed:output:0 zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
zoom_matrix/zeros_1/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:^
zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
zoom_matrix/zeros_1Fill#zoom_matrix/zeros_1/packed:output:0"zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
!zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_4StridedSliceconcat:output:0*zoom_matrix/strided_slice_4/stack:output:0,zoom_matrix/strided_slice_4/stack_1:output:0,zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask^
zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
zoom_matrix/zeros_2/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:^
zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
zoom_matrix/zeros_2Fill#zoom_matrix/zeros_2/packed:output:0"zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ë
zoom_matrix/concatConcatV2$zoom_matrix/strided_slice_3:output:0zoom_matrix/zeros:output:0zoom_matrix/mul:z:0zoom_matrix/zeros_1:output:0$zoom_matrix/strided_slice_4:output:0zoom_matrix/mul_1:z:0zoom_matrix/zeros_2:output:0 zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
transform/ShapeShapeinputs*
T0*
_output_shapes
:g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputszoom_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´h
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ´´: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
õ
d
H__inference_sequential_12_layer_call_and_return_conditional_losses_44795

inputs
identityÎ
random_flip_6/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_random_flip_6_layer_call_and_return_conditional_losses_44780ö
!random_rotation_6/PartitionedCallPartitionedCall&random_flip_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_random_rotation_6_layer_call_and_return_conditional_losses_44786ò
random_zoom_6/PartitionedCallPartitionedCall*random_rotation_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_random_zoom_6_layer_call_and_return_conditional_losses_44792x
IdentityIdentity&random_zoom_6/PartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´´:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
Ø
M
1__inference_random_rotation_6_layer_call_fn_46721

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_random_rotation_6_layer_call_and_return_conditional_losses_44786j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´´:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
÷L
¦
__inference__traced_save_47114
file_prefix/
+savev2_conv2d_22_kernel_read_readvariableop-
)savev2_conv2d_22_bias_read_readvariableop/
+savev2_conv2d_23_kernel_read_readvariableop-
)savev2_conv2d_23_bias_read_readvariableop/
+savev2_conv2d_24_kernel_read_readvariableop-
)savev2_conv2d_24_bias_read_readvariableop/
+savev2_conv2d_25_kernel_read_readvariableop-
)savev2_conv2d_25_bias_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_random_flip_6_statevar_read_readvariableop	9
5savev2_random_rotation_6_statevar_read_readvariableop	5
1savev2_random_zoom_6_statevar_read_readvariableop	;
7savev2_rmsprop_conv2d_22_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_22_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_23_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_23_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_24_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_24_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_25_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_25_bias_rms_read_readvariableop:
6savev2_rmsprop_dense_12_kernel_rms_read_readvariableop8
4savev2_rmsprop_dense_12_bias_rms_read_readvariableop:
6savev2_rmsprop_dense_13_kernel_rms_read_readvariableop8
4savev2_rmsprop_dense_13_bias_rms_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ò
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*û
valueñBî%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer-0/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-0/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-0/layer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH·
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ÿ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_22_kernel_read_readvariableop)savev2_conv2d_22_bias_read_readvariableop+savev2_conv2d_23_kernel_read_readvariableop)savev2_conv2d_23_bias_read_readvariableop+savev2_conv2d_24_kernel_read_readvariableop)savev2_conv2d_24_bias_read_readvariableop+savev2_conv2d_25_kernel_read_readvariableop)savev2_conv2d_25_bias_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_random_flip_6_statevar_read_readvariableop5savev2_random_rotation_6_statevar_read_readvariableop1savev2_random_zoom_6_statevar_read_readvariableop7savev2_rmsprop_conv2d_22_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_22_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_23_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_23_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_24_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_24_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_25_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_25_bias_rms_read_readvariableop6savev2_rmsprop_dense_12_kernel_rms_read_readvariableop4savev2_rmsprop_dense_12_bias_rms_read_readvariableop6savev2_rmsprop_dense_13_kernel_rms_read_readvariableop4savev2_rmsprop_dense_13_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *3
dtypes)
'2%				
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Å
_input_shapes³
°: ::: : : @:@:@@:@:
À(::	:: : : : : : : : : :::::: : : @:@:@@:@:
À(::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:&	"
 
_output_shapes
:
À(:!


_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@:  

_output_shapes
:@:&!"
 
_output_shapes
:
À(:!"

_output_shapes	
::%#!

_output_shapes
:	: $

_output_shapes
::%

_output_shapes
: 

ý
D__inference_conv2d_22_layer_call_and_return_conditional_losses_45252

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²²*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²²Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²²k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²²w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ´´: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs

Å
L__inference_random_rotation_6_layer_call_and_return_conditional_losses_46850

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity¢stateful_uniform/RngReadAndSkip;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿj
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿa
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿj
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: d
stateful_uniform/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ¿Y
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ?`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ¶
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¢
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?g
rotation_matrix/subSub
Cast_1:y:0rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: ^
rotation_matrix/CosCosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: |
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?i
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: ~
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?i
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: `
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?i
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
:m
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
rotation_matrix/strided_sliceStridedSlicerotation_matrix/Shape:output:0,rotation_matrix/strided_slice/stack:output:0.rotation_matrix/strided_slice/stack_1:output:0.rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
rotation_matrix/Cos_2Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_maskv
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_maskv
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ø
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :£
rotation_matrix/zeros/packedPack&rotation_matrix/strided_slice:output:0'rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
transform/ShapeShapeinputs*
T0*
_output_shapes
:g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ¡
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´h
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ´´: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs

q
H__inference_sequential_12_layer_call_and_return_conditional_losses_45162
random_flip_6_input
identityÛ
random_flip_6/PartitionedCallPartitionedCallrandom_flip_6_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_random_flip_6_layer_call_and_return_conditional_losses_44780ö
!random_rotation_6/PartitionedCallPartitionedCall&random_flip_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_random_rotation_6_layer_call_and_return_conditional_losses_44786ò
random_zoom_6/PartitionedCallPartitionedCall*random_rotation_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_random_zoom_6_layer_call_and_return_conditional_losses_44792x
IdentityIdentity&random_zoom_6/PartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´´:f b
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
-
_user_specified_namerandom_flip_6_input
»
Ð
H__inference_sequential_12_layer_call_and_return_conditional_losses_45135

inputs!
random_flip_6_45125:	%
random_rotation_6_45128:	!
random_zoom_6_45131:	
identity¢%random_flip_6/StatefulPartitionedCall¢)random_rotation_6/StatefulPartitionedCall¢%random_zoom_6/StatefulPartitionedCallô
%random_flip_6/StatefulPartitionedCallStatefulPartitionedCallinputsrandom_flip_6_45125*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_random_flip_6_layer_call_and_return_conditional_losses_45110¨
)random_rotation_6/StatefulPartitionedCallStatefulPartitionedCall.random_flip_6/StatefulPartitionedCall:output:0random_rotation_6_45128*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_random_rotation_6_layer_call_and_return_conditional_losses_45039 
%random_zoom_6/StatefulPartitionedCallStatefulPartitionedCall2random_rotation_6/StatefulPartitionedCall:output:0random_zoom_6_45131*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_random_zoom_6_layer_call_and_return_conditional_losses_44908
IdentityIdentity.random_zoom_6/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´Â
NoOpNoOp&^random_flip_6/StatefulPartitionedCall*^random_rotation_6/StatefulPartitionedCall&^random_zoom_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ´´: : : 2N
%random_flip_6/StatefulPartitionedCall%random_flip_6/StatefulPartitionedCall2V
)random_rotation_6/StatefulPartitionedCall)random_rotation_6/StatefulPartitionedCall2N
%random_zoom_6/StatefulPartitionedCall%random_zoom_6/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs

Ã
!__inference__traced_restore_47232
file_prefix;
!assignvariableop_conv2d_22_kernel:/
!assignvariableop_1_conv2d_22_bias:=
#assignvariableop_2_conv2d_23_kernel: /
!assignvariableop_3_conv2d_23_bias: =
#assignvariableop_4_conv2d_24_kernel: @/
!assignvariableop_5_conv2d_24_bias:@=
#assignvariableop_6_conv2d_25_kernel:@@/
!assignvariableop_7_conv2d_25_bias:@6
"assignvariableop_8_dense_12_kernel:
À(/
 assignvariableop_9_dense_12_bias:	6
#assignvariableop_10_dense_13_kernel:	/
!assignvariableop_11_dense_13_bias:*
 assignvariableop_12_rmsprop_iter:	 +
!assignvariableop_13_rmsprop_decay: 3
)assignvariableop_14_rmsprop_learning_rate: .
$assignvariableop_15_rmsprop_momentum: )
assignvariableop_16_rmsprop_rho: #
assignvariableop_17_total: #
assignvariableop_18_count: %
assignvariableop_19_total_1: %
assignvariableop_20_count_1: 8
*assignvariableop_21_random_flip_6_statevar:	<
.assignvariableop_22_random_rotation_6_statevar:	8
*assignvariableop_23_random_zoom_6_statevar:	J
0assignvariableop_24_rmsprop_conv2d_22_kernel_rms:<
.assignvariableop_25_rmsprop_conv2d_22_bias_rms:J
0assignvariableop_26_rmsprop_conv2d_23_kernel_rms: <
.assignvariableop_27_rmsprop_conv2d_23_bias_rms: J
0assignvariableop_28_rmsprop_conv2d_24_kernel_rms: @<
.assignvariableop_29_rmsprop_conv2d_24_bias_rms:@J
0assignvariableop_30_rmsprop_conv2d_25_kernel_rms:@@<
.assignvariableop_31_rmsprop_conv2d_25_bias_rms:@C
/assignvariableop_32_rmsprop_dense_12_kernel_rms:
À(<
-assignvariableop_33_rmsprop_dense_12_bias_rms:	B
/assignvariableop_34_rmsprop_dense_13_kernel_rms:	;
-assignvariableop_35_rmsprop_dense_13_bias_rms:
identity_37¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Õ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*û
valueñBî%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer-0/layer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-0/layer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBRlayer-0/layer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHº
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ú
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ª
_output_shapes
:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%				[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_22_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_22_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_23_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_23_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_24_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_24_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_25_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_25_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_12_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_12_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_13_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_13_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_12AssignVariableOp assignvariableop_12_rmsprop_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp!assignvariableop_13_rmsprop_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp)assignvariableop_14_rmsprop_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp$assignvariableop_15_rmsprop_momentumIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_rmsprop_rhoIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_21AssignVariableOp*assignvariableop_21_random_flip_6_statevarIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_22AssignVariableOp.assignvariableop_22_random_rotation_6_statevarIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_23AssignVariableOp*assignvariableop_23_random_zoom_6_statevarIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_24AssignVariableOp0assignvariableop_24_rmsprop_conv2d_22_kernel_rmsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp.assignvariableop_25_rmsprop_conv2d_22_bias_rmsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_26AssignVariableOp0assignvariableop_26_rmsprop_conv2d_23_kernel_rmsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp.assignvariableop_27_rmsprop_conv2d_23_bias_rmsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_28AssignVariableOp0assignvariableop_28_rmsprop_conv2d_24_kernel_rmsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp.assignvariableop_29_rmsprop_conv2d_24_bias_rmsIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_30AssignVariableOp0assignvariableop_30_rmsprop_conv2d_25_kernel_rmsIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp.assignvariableop_31_rmsprop_conv2d_25_bias_rmsIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_32AssignVariableOp/assignvariableop_32_rmsprop_dense_12_kernel_rmsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp-assignvariableop_33_rmsprop_dense_12_bias_rmsIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_34AssignVariableOp/assignvariableop_34_rmsprop_dense_13_kernel_rmsIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp-assignvariableop_35_rmsprop_dense_13_bias_rmsIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ç
Identity_36Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_37IdentityIdentity_36:output:0^NoOp_1*
T0*
_output_shapes
: Ô
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_37Identity_37:output:0*]
_input_shapesL
J: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
õ

)__inference_conv2d_22_layer_call_fn_46480

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²²*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_45252y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²²`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ´´: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
»
L
0__inference_max_pooling2d_25_layer_call_fn_46586

inputs
identityÜ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_45220
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_45220

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_45184

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_45208

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
Ý
H__inference_sequential_12_layer_call_and_return_conditional_losses_45175
random_flip_6_input!
random_flip_6_45165:	%
random_rotation_6_45168:	!
random_zoom_6_45171:	
identity¢%random_flip_6/StatefulPartitionedCall¢)random_rotation_6/StatefulPartitionedCall¢%random_zoom_6/StatefulPartitionedCall
%random_flip_6/StatefulPartitionedCallStatefulPartitionedCallrandom_flip_6_inputrandom_flip_6_45165*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_random_flip_6_layer_call_and_return_conditional_losses_45110¨
)random_rotation_6/StatefulPartitionedCallStatefulPartitionedCall.random_flip_6/StatefulPartitionedCall:output:0random_rotation_6_45168*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_random_rotation_6_layer_call_and_return_conditional_losses_45039 
%random_zoom_6/StatefulPartitionedCallStatefulPartitionedCall2random_rotation_6/StatefulPartitionedCall:output:0random_zoom_6_45171*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_random_zoom_6_layer_call_and_return_conditional_losses_44908
IdentityIdentity.random_zoom_6/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´Â
NoOpNoOp&^random_flip_6/StatefulPartitionedCall*^random_rotation_6/StatefulPartitionedCall&^random_zoom_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ´´: : : 2N
%random_flip_6/StatefulPartitionedCall%random_flip_6/StatefulPartitionedCall2V
)random_rotation_6/StatefulPartitionedCall)random_rotation_6/StatefulPartitionedCall2N
%random_zoom_6/StatefulPartitionedCall%random_zoom_6/StatefulPartitionedCall:f b
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
-
_user_specified_namerandom_flip_6_input
ø
b
F__inference_rescaling_6_layer_call_and_return_conditional_losses_46471

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    _
mulMulinputsCast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´d
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´Y
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´´:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
«
Í
-__inference_sequential_13_layer_call_fn_45724

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
À(
	unknown_8:	
	unknown_9:	

unknown_10:
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_13_layer_call_and_return_conditional_losses_45356o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ´´: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
»
L
0__inference_max_pooling2d_23_layer_call_fn_46526

inputs
identityÜ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_45196
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ý
D__inference_conv2d_24_layer_call_and_return_conditional_losses_45288

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ++ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ++ 
 
_user_specified_nameinputs
í

)__inference_conv2d_25_layer_call_fn_46570

inputs!
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_45306w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

h
L__inference_random_rotation_6_layer_call_and_return_conditional_losses_46732

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´´:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_46531

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
úS

 __inference__wrapped_model_44769
input_7P
6sequential_13_conv2d_22_conv2d_readvariableop_resource:E
7sequential_13_conv2d_22_biasadd_readvariableop_resource:P
6sequential_13_conv2d_23_conv2d_readvariableop_resource: E
7sequential_13_conv2d_23_biasadd_readvariableop_resource: P
6sequential_13_conv2d_24_conv2d_readvariableop_resource: @E
7sequential_13_conv2d_24_biasadd_readvariableop_resource:@P
6sequential_13_conv2d_25_conv2d_readvariableop_resource:@@E
7sequential_13_conv2d_25_biasadd_readvariableop_resource:@I
5sequential_13_dense_12_matmul_readvariableop_resource:
À(E
6sequential_13_dense_12_biasadd_readvariableop_resource:	H
5sequential_13_dense_13_matmul_readvariableop_resource:	D
6sequential_13_dense_13_biasadd_readvariableop_resource:
identity¢.sequential_13/conv2d_22/BiasAdd/ReadVariableOp¢-sequential_13/conv2d_22/Conv2D/ReadVariableOp¢.sequential_13/conv2d_23/BiasAdd/ReadVariableOp¢-sequential_13/conv2d_23/Conv2D/ReadVariableOp¢.sequential_13/conv2d_24/BiasAdd/ReadVariableOp¢-sequential_13/conv2d_24/Conv2D/ReadVariableOp¢.sequential_13/conv2d_25/BiasAdd/ReadVariableOp¢-sequential_13/conv2d_25/Conv2D/ReadVariableOp¢-sequential_13/dense_12/BiasAdd/ReadVariableOp¢,sequential_13/dense_12/MatMul/ReadVariableOp¢-sequential_13/dense_13/BiasAdd/ReadVariableOp¢,sequential_13/dense_13/MatMul/ReadVariableOpe
 sequential_13/rescaling_6/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;g
"sequential_13/rescaling_6/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    
sequential_13/rescaling_6/mulMulinput_7)sequential_13/rescaling_6/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´²
sequential_13/rescaling_6/addAddV2!sequential_13/rescaling_6/mul:z:0+sequential_13/rescaling_6/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´¬
-sequential_13/conv2d_22/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ç
sequential_13/conv2d_22/Conv2DConv2D!sequential_13/rescaling_6/add:z:05sequential_13/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²²*
paddingVALID*
strides
¢
.sequential_13/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ç
sequential_13/conv2d_22/BiasAddBiasAdd'sequential_13/conv2d_22/Conv2D:output:06sequential_13/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²²
sequential_13/conv2d_22/ReluRelu(sequential_13/conv2d_22/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²²Ê
&sequential_13/max_pooling2d_22/MaxPoolMaxPool*sequential_13/conv2d_22/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿYY*
ksize
*
paddingVALID*
strides
¬
-sequential_13/conv2d_23/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ó
sequential_13/conv2d_23/Conv2DConv2D/sequential_13/max_pooling2d_22/MaxPool:output:05sequential_13/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿWW *
paddingVALID*
strides
¢
.sequential_13/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Å
sequential_13/conv2d_23/BiasAddBiasAdd'sequential_13/conv2d_23/Conv2D:output:06sequential_13/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿWW 
sequential_13/conv2d_23/ReluRelu(sequential_13/conv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿWW Ê
&sequential_13/max_pooling2d_23/MaxPoolMaxPool*sequential_13/conv2d_23/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ++ *
ksize
*
paddingVALID*
strides
¬
-sequential_13/conv2d_24/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0ó
sequential_13/conv2d_24/Conv2DConv2D/sequential_13/max_pooling2d_23/MaxPool:output:05sequential_13/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@*
paddingVALID*
strides
¢
.sequential_13/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Å
sequential_13/conv2d_24/BiasAddBiasAdd'sequential_13/conv2d_24/Conv2D:output:06sequential_13/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@
sequential_13/conv2d_24/ReluRelu(sequential_13/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@Ê
&sequential_13/max_pooling2d_24/MaxPoolMaxPool*sequential_13/conv2d_24/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
¬
-sequential_13/conv2d_25/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0ó
sequential_13/conv2d_25/Conv2DConv2D/sequential_13/max_pooling2d_24/MaxPool:output:05sequential_13/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
¢
.sequential_13/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Å
sequential_13/conv2d_25/BiasAddBiasAdd'sequential_13/conv2d_25/Conv2D:output:06sequential_13/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
sequential_13/conv2d_25/ReluRelu(sequential_13/conv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ê
&sequential_13/max_pooling2d_25/MaxPoolMaxPool*sequential_13/conv2d_25/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@*
ksize
*
paddingVALID*
strides
n
sequential_13/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  ¶
sequential_13/flatten_6/ReshapeReshape/sequential_13/max_pooling2d_25/MaxPool:output:0&sequential_13/flatten_6/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ(¤
,sequential_13/dense_12/MatMul/ReadVariableOpReadVariableOp5sequential_13_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
À(*
dtype0º
sequential_13/dense_12/MatMulMatMul(sequential_13/flatten_6/Reshape:output:04sequential_13/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
-sequential_13/dense_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¼
sequential_13/dense_12/BiasAddBiasAdd'sequential_13/dense_12/MatMul:product:05sequential_13/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_13/dense_12/ReluRelu'sequential_13/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
,sequential_13/dense_13/MatMul/ReadVariableOpReadVariableOp5sequential_13_dense_13_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0º
sequential_13/dense_13/MatMulMatMul)sequential_13/dense_12/Relu:activations:04sequential_13/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-sequential_13/dense_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0»
sequential_13/dense_13/BiasAddBiasAdd'sequential_13/dense_13/MatMul:product:05sequential_13/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_13/dense_13/SoftmaxSoftmax'sequential_13/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity(sequential_13/dense_13/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp/^sequential_13/conv2d_22/BiasAdd/ReadVariableOp.^sequential_13/conv2d_22/Conv2D/ReadVariableOp/^sequential_13/conv2d_23/BiasAdd/ReadVariableOp.^sequential_13/conv2d_23/Conv2D/ReadVariableOp/^sequential_13/conv2d_24/BiasAdd/ReadVariableOp.^sequential_13/conv2d_24/Conv2D/ReadVariableOp/^sequential_13/conv2d_25/BiasAdd/ReadVariableOp.^sequential_13/conv2d_25/Conv2D/ReadVariableOp.^sequential_13/dense_12/BiasAdd/ReadVariableOp-^sequential_13/dense_12/MatMul/ReadVariableOp.^sequential_13/dense_13/BiasAdd/ReadVariableOp-^sequential_13/dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ´´: : : : : : : : : : : : 2`
.sequential_13/conv2d_22/BiasAdd/ReadVariableOp.sequential_13/conv2d_22/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_22/Conv2D/ReadVariableOp-sequential_13/conv2d_22/Conv2D/ReadVariableOp2`
.sequential_13/conv2d_23/BiasAdd/ReadVariableOp.sequential_13/conv2d_23/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_23/Conv2D/ReadVariableOp-sequential_13/conv2d_23/Conv2D/ReadVariableOp2`
.sequential_13/conv2d_24/BiasAdd/ReadVariableOp.sequential_13/conv2d_24/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_24/Conv2D/ReadVariableOp-sequential_13/conv2d_24/Conv2D/ReadVariableOp2`
.sequential_13/conv2d_25/BiasAdd/ReadVariableOp.sequential_13/conv2d_25/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_25/Conv2D/ReadVariableOp-sequential_13/conv2d_25/Conv2D/ReadVariableOp2^
-sequential_13/dense_12/BiasAdd/ReadVariableOp-sequential_13/dense_12/BiasAdd/ReadVariableOp2\
,sequential_13/dense_12/MatMul/ReadVariableOp,sequential_13/dense_12/MatMul/ReadVariableOp2^
-sequential_13/dense_13/BiasAdd/ReadVariableOp-sequential_13/dense_13/BiasAdd/ReadVariableOp2\
,sequential_13/dense_13/MatMul/ReadVariableOp,sequential_13/dense_13/MatMul/ReadVariableOp:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
!
_user_specified_name	input_7
n
Á
H__inference_random_zoom_6_layer_call_and_return_conditional_losses_44908

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity¢stateful_uniform/RngReadAndSkip;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿj
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿa
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿj
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: Z
stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
stateful_uniform/shapePackstrided_slice:output:0!stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?Y
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ¶
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¢
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2stateful_uniform:z:0stateful_uniform:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
zoom_matrix/ShapeShapeconcat:output:0*
T0*
_output_shapes
:i
zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
zoom_matrix/strided_sliceStridedSlicezoom_matrix/Shape:output:0(zoom_matrix/strided_slice/stack:output:0*zoom_matrix/strided_slice/stack_1:output:0*zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?_
zoom_matrix/subSub
Cast_1:y:0zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: Z
zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @t
zoom_matrix/truedivRealDivzoom_matrix/sub:z:0zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: v
!zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            x
#zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_1StridedSliceconcat:output:0*zoom_matrix/strided_slice_1/stack:output:0,zoom_matrix/strided_slice_1/stack_1:output:0,zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskX
zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
zoom_matrix/sub_1Subzoom_matrix/sub_1/x:output:0$zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
zoom_matrix/mulMulzoom_matrix/truediv:z:0zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
zoom_matrix/sub_2SubCast:y:0zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: \
zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @z
zoom_matrix/truediv_1RealDivzoom_matrix/sub_2:z:0 zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: v
!zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_2StridedSliceconcat:output:0*zoom_matrix/strided_slice_2/stack:output:0,zoom_matrix/strided_slice_2/stack_1:output:0,zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskX
zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
zoom_matrix/sub_3Subzoom_matrix/sub_3/x:output:0$zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
zoom_matrix/mul_1Mulzoom_matrix/truediv_1:z:0zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
!zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            x
#zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_3StridedSliceconcat:output:0*zoom_matrix/strided_slice_3/stack:output:0,zoom_matrix/strided_slice_3/stack_1:output:0,zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask\
zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
zoom_matrix/zeros/packedPack"zoom_matrix/strided_slice:output:0#zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:\
zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
zoom_matrix/zerosFill!zoom_matrix/zeros/packed:output:0 zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
zoom_matrix/zeros_1/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:^
zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
zoom_matrix/zeros_1Fill#zoom_matrix/zeros_1/packed:output:0"zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
!zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           x
#zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         Ò
zoom_matrix/strided_slice_4StridedSliceconcat:output:0*zoom_matrix/strided_slice_4/stack:output:0,zoom_matrix/strided_slice_4/stack_1:output:0,zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask^
zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
zoom_matrix/zeros_2/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:^
zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
zoom_matrix/zeros_2Fill#zoom_matrix/zeros_2/packed:output:0"zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ë
zoom_matrix/concatConcatV2$zoom_matrix/strided_slice_3:output:0zoom_matrix/zeros:output:0zoom_matrix/mul:z:0zoom_matrix/zeros_1:output:0$zoom_matrix/strided_slice_4:output:0zoom_matrix/mul_1:z:0zoom_matrix/zeros_2:output:0 zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
transform/ShapeShapeinputs*
T0*
_output_shapes
:g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputszoom_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´h
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ´´: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
»
L
0__inference_max_pooling2d_22_layer_call_fn_46496

inputs
identityÜ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_45184
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

d
H__inference_random_flip_6_layer_call_and_return_conditional_losses_46658

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´´:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
Ð
I
-__inference_random_flip_6_layer_call_fn_46647

inputs
identityÀ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_random_flip_6_layer_call_and_return_conditional_losses_44780j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´´:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_46591

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í

)__inference_conv2d_23_layer_call_fn_46510

inputs!
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿWW *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_45270w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿWW `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿYY: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿYY
 
_user_specified_nameinputs
Æ

(__inference_dense_13_layer_call_fn_46631

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_45349o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î

1__inference_random_rotation_6_layer_call_fn_46728

inputs
unknown:	
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_random_rotation_6_layer_call_and_return_conditional_losses_45039y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ´´: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
Å
}
-__inference_random_zoom_6_layer_call_fn_46862

inputs
unknown:	
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_random_zoom_6_layer_call_and_return_conditional_losses_44908y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ´´: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
¸8
ú
H__inference_sequential_13_layer_call_and_return_conditional_losses_45533

inputs!
sequential_12_45489:	!
sequential_12_45491:	!
sequential_12_45493:	)
conv2d_22_45497:
conv2d_22_45499:)
conv2d_23_45503: 
conv2d_23_45505: )
conv2d_24_45509: @
conv2d_24_45511:@)
conv2d_25_45515:@@
conv2d_25_45517:@"
dense_12_45522:
À(
dense_12_45524:	!
dense_13_45527:	
dense_13_45529:
identity¢!conv2d_22/StatefulPartitionedCall¢!conv2d_23/StatefulPartitionedCall¢!conv2d_24/StatefulPartitionedCall¢!conv2d_25/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall¢%sequential_12/StatefulPartitionedCall 
%sequential_12/StatefulPartitionedCallStatefulPartitionedCallinputssequential_12_45489sequential_12_45491sequential_12_45493*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_12_layer_call_and_return_conditional_losses_45135ò
rescaling_6/PartitionedCallPartitionedCall.sequential_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_rescaling_6_layer_call_and_return_conditional_losses_45239
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall$rescaling_6/PartitionedCall:output:0conv2d_22_45497conv2d_22_45499*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²²*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_45252ö
 max_pooling2d_22/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿYY* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_45184
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_22/PartitionedCall:output:0conv2d_23_45503conv2d_23_45505*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿWW *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_45270ö
 max_pooling2d_23/PartitionedCallPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ++ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_45196
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_23/PartitionedCall:output:0conv2d_24_45509conv2d_24_45511*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ))@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_45288ö
 max_pooling2d_24/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_45208
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_24/PartitionedCall:output:0conv2d_25_45515conv2d_25_45517*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_45306ö
 max_pooling2d_25/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_45220à
flatten_6/PartitionedCallPartitionedCall)max_pooling2d_25/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_6_layer_call_and_return_conditional_losses_45319
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_12_45522dense_12_45524*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_45332
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_45527dense_13_45529*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_13_layer_call_and_return_conditional_losses_45349x
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
NoOpNoOp"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall&^sequential_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ´´: : : : : : : : : : : : : : : 2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2N
%sequential_12/StatefulPartitionedCall%sequential_12/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
Æ
`
D__inference_flatten_6_layer_call_and_return_conditional_losses_46602

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ(Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ		@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@
 
_user_specified_nameinputs

d
H__inference_random_zoom_6_layer_call_and_return_conditional_losses_46866

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´´:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
Ê

(__inference_dense_12_layer_call_fn_46611

inputs
unknown:
À(
	unknown_0:	
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_12_layer_call_and_return_conditional_losses_45332p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ(: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ(
 
_user_specified_nameinputs
Ð
I
-__inference_sequential_12_layer_call_fn_46173

inputs
identityÀ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_12_layer_call_and_return_conditional_losses_44795j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´´:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
Ì
G
+__inference_rescaling_6_layer_call_fn_46463

inputs
identity¾
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_rescaling_6_layer_call_and_return_conditional_losses_45239j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´´:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs

Å
L__inference_random_rotation_6_layer_call_and_return_conditional_losses_45039

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity¢stateful_uniform/RngReadAndSkip;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿj
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿa
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿj
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: d
stateful_uniform/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:Y
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ¿Y
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ?`
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: Y
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :o
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ¶
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:n
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¬
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask}
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0p
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¢
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?g
rotation_matrix/subSub
Cast_1:y:0rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: ^
rotation_matrix/CosCosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: |
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?i
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: ~
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?i
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: `
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?i
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
:m
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
rotation_matrix/strided_sliceStridedSlicerotation_matrix/Shape:output:0,rotation_matrix/strided_slice/stack:output:0.rotation_matrix/strided_slice/stack_1:output:0.rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
rotation_matrix/Cos_2Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_maskv
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_maskv
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ø
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask`
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :£
rotation_matrix/zeros/packedPack&rotation_matrix/strided_slice:output:0'rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿE
transform/ShapeShapeinputs*
T0*
_output_shapes
:g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Y
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ¡
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´h
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ´´: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
ÝM
Ó
H__inference_random_flip_6_layer_call_and_return_conditional_losses_45110

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity¢(stateful_uniform_full_int/RngReadAndSkipi
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: b
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Ú
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:w
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0y
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ï
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0_
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	T

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R x
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask
3stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
&stateless_random_flip_left_right/ShapeShape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:~
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ö
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask®
?stateless_random_flip_left_right/stateless_random_uniform/shapePack7stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :þ
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0_stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿõ
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿû
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :r
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Î
.stateless_random_flip_left_right/Reshape/shapePack7stateless_random_flip_left_right/strided_slice:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:å
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:ë
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´È
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´k
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Â
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´¿
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
IdentityIdentity(stateless_random_flip_left_right/add:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´q
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ´´: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
ùý
ã
H__inference_sequential_12_layer_call_and_return_conditional_losses_46458

inputsM
?random_flip_6_stateful_uniform_full_int_rngreadandskip_resource:	H
:random_rotation_6_stateful_uniform_rngreadandskip_resource:	D
6random_zoom_6_stateful_uniform_rngreadandskip_resource:	
identity¢6random_flip_6/stateful_uniform_full_int/RngReadAndSkip¢1random_rotation_6/stateful_uniform/RngReadAndSkip¢-random_zoom_6/stateful_uniform/RngReadAndSkipw
-random_flip_6/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:w
-random_flip_6/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: Å
,random_flip_6/stateful_uniform_full_int/ProdProd6random_flip_6/stateful_uniform_full_int/shape:output:06random_flip_6/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: p
.random_flip_6/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
.random_flip_6/stateful_uniform_full_int/Cast_1Cast5random_flip_6/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 
6random_flip_6/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip?random_flip_6_stateful_uniform_full_int_rngreadandskip_resource7random_flip_6/stateful_uniform_full_int/Cast/x:output:02random_flip_6/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:
;random_flip_6/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=random_flip_6/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=random_flip_6/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5random_flip_6/stateful_uniform_full_int/strided_sliceStridedSlice>random_flip_6/stateful_uniform_full_int/RngReadAndSkip:value:0Drandom_flip_6/stateful_uniform_full_int/strided_slice/stack:output:0Frandom_flip_6/stateful_uniform_full_int/strided_slice/stack_1:output:0Frandom_flip_6/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask«
/random_flip_6/stateful_uniform_full_int/BitcastBitcast>random_flip_6/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
=random_flip_6/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?random_flip_6/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?random_flip_6/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
7random_flip_6/stateful_uniform_full_int/strided_slice_1StridedSlice>random_flip_6/stateful_uniform_full_int/RngReadAndSkip:value:0Frandom_flip_6/stateful_uniform_full_int/strided_slice_1/stack:output:0Hrandom_flip_6/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Hrandom_flip_6/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:¯
1random_flip_6/stateful_uniform_full_int/Bitcast_1Bitcast@random_flip_6/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0m
+random_flip_6/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :×
'random_flip_6/stateful_uniform_full_intStatelessRandomUniformFullIntV26random_flip_6/stateful_uniform_full_int/shape:output:0:random_flip_6/stateful_uniform_full_int/Bitcast_1:output:08random_flip_6/stateful_uniform_full_int/Bitcast:output:04random_flip_6/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	b
random_flip_6/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R ¢
random_flip_6/stackPack0random_flip_6/stateful_uniform_full_int:output:0!random_flip_6/zeros_like:output:0*
N*
T0	*
_output_shapes

:r
!random_flip_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        t
#random_flip_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       t
#random_flip_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ½
random_flip_6/strided_sliceStridedSlicerandom_flip_6/stack:output:0*random_flip_6/strided_slice/stack:output:0,random_flip_6/strided_slice/stack_1:output:0,random_flip_6/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask¬
Arandom_flip_6/stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´®
4random_flip_6/stateless_random_flip_left_right/ShapeShapeJrandom_flip_6/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:
Brandom_flip_6/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Drandom_flip_6/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Drandom_flip_6/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¼
<random_flip_6/stateless_random_flip_left_right/strided_sliceStridedSlice=random_flip_6/stateless_random_flip_left_right/Shape:output:0Krandom_flip_6/stateless_random_flip_left_right/strided_slice/stack:output:0Mrandom_flip_6/stateless_random_flip_left_right/strided_slice/stack_1:output:0Mrandom_flip_6/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÊ
Mrandom_flip_6/stateless_random_flip_left_right/stateless_random_uniform/shapePackErandom_flip_6/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:
Krandom_flip_6/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
Krandom_flip_6/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ì
drandom_flip_6/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter$random_flip_6/strided_slice:output:0* 
_output_shapes
::¦
drandom_flip_6/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :Ä
`random_flip_6/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Vrandom_flip_6/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0jrandom_flip_6/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0nrandom_flip_6/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0mrandom_flip_6/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Krandom_flip_6/stateless_random_flip_left_right/stateless_random_uniform/subSubTrandom_flip_6/stateless_random_flip_left_right/stateless_random_uniform/max:output:0Trandom_flip_6/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ¼
Krandom_flip_6/stateless_random_flip_left_right/stateless_random_uniform/mulMulirandom_flip_6/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Orandom_flip_6/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
Grandom_flip_6/stateless_random_flip_left_right/stateless_random_uniformAddV2Orandom_flip_6/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0Trandom_flip_6/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>random_flip_6/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
>random_flip_6/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
>random_flip_6/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
<random_flip_6/stateless_random_flip_left_right/Reshape/shapePackErandom_flip_6/stateless_random_flip_left_right/strided_slice:output:0Grandom_flip_6/stateless_random_flip_left_right/Reshape/shape/1:output:0Grandom_flip_6/stateless_random_flip_left_right/Reshape/shape/2:output:0Grandom_flip_6/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
6random_flip_6/stateless_random_flip_left_right/ReshapeReshapeKrandom_flip_6/stateless_random_flip_left_right/stateless_random_uniform:z:0Erandom_flip_6/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
4random_flip_6/stateless_random_flip_left_right/RoundRound?random_flip_6/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
=random_flip_6/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:
8random_flip_6/stateless_random_flip_left_right/ReverseV2	ReverseV2Jrandom_flip_6/stateless_random_flip_left_right/control_dependency:output:0Frandom_flip_6/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´ò
2random_flip_6/stateless_random_flip_left_right/mulMul8random_flip_6/stateless_random_flip_left_right/Round:y:0Arandom_flip_6/stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´y
4random_flip_6/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ì
2random_flip_6/stateless_random_flip_left_right/subSub=random_flip_6/stateless_random_flip_left_right/sub/x:output:08random_flip_6/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿû
4random_flip_6/stateless_random_flip_left_right/mul_1Mul6random_flip_6/stateless_random_flip_left_right/sub:z:0Jrandom_flip_6/stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´é
2random_flip_6/stateless_random_flip_left_right/addAddV26random_flip_6/stateless_random_flip_left_right/mul:z:08random_flip_6/stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´}
random_rotation_6/ShapeShape6random_flip_6/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
:o
%random_rotation_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'random_rotation_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'random_rotation_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:«
random_rotation_6/strided_sliceStridedSlice random_rotation_6/Shape:output:0.random_rotation_6/strided_slice/stack:output:00random_rotation_6/strided_slice/stack_1:output:00random_rotation_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
'random_rotation_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ|
)random_rotation_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿs
)random_rotation_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:³
!random_rotation_6/strided_slice_1StridedSlice random_rotation_6/Shape:output:00random_rotation_6/strided_slice_1/stack:output:02random_rotation_6/strided_slice_1/stack_1:output:02random_rotation_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
random_rotation_6/CastCast*random_rotation_6/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: z
'random_rotation_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ|
)random_rotation_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿs
)random_rotation_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:³
!random_rotation_6/strided_slice_2StridedSlice random_rotation_6/Shape:output:00random_rotation_6/strided_slice_2/stack:output:02random_rotation_6/strided_slice_2/stack_1:output:02random_rotation_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
random_rotation_6/Cast_1Cast*random_rotation_6/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 
(random_rotation_6/stateful_uniform/shapePack(random_rotation_6/strided_slice:output:0*
N*
T0*
_output_shapes
:k
&random_rotation_6/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ¿k
&random_rotation_6/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ?r
(random_rotation_6/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¶
'random_rotation_6/stateful_uniform/ProdProd1random_rotation_6/stateful_uniform/shape:output:01random_rotation_6/stateful_uniform/Const:output:0*
T0*
_output_shapes
: k
)random_rotation_6/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
)random_rotation_6/stateful_uniform/Cast_1Cast0random_rotation_6/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: þ
1random_rotation_6/stateful_uniform/RngReadAndSkipRngReadAndSkip:random_rotation_6_stateful_uniform_rngreadandskip_resource2random_rotation_6/stateful_uniform/Cast/x:output:0-random_rotation_6/stateful_uniform/Cast_1:y:0*
_output_shapes
:
6random_rotation_6/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8random_rotation_6/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8random_rotation_6/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
0random_rotation_6/stateful_uniform/strided_sliceStridedSlice9random_rotation_6/stateful_uniform/RngReadAndSkip:value:0?random_rotation_6/stateful_uniform/strided_slice/stack:output:0Arandom_rotation_6/stateful_uniform/strided_slice/stack_1:output:0Arandom_rotation_6/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask¡
*random_rotation_6/stateful_uniform/BitcastBitcast9random_rotation_6/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
8random_rotation_6/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
:random_rotation_6/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:random_rotation_6/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ü
2random_rotation_6/stateful_uniform/strided_slice_1StridedSlice9random_rotation_6/stateful_uniform/RngReadAndSkip:value:0Arandom_rotation_6/stateful_uniform/strided_slice_1/stack:output:0Crandom_rotation_6/stateful_uniform/strided_slice_1/stack_1:output:0Crandom_rotation_6/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:¥
,random_rotation_6/stateful_uniform/Bitcast_1Bitcast;random_rotation_6/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0
?random_rotation_6/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :å
;random_rotation_6/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV21random_rotation_6/stateful_uniform/shape:output:05random_rotation_6/stateful_uniform/Bitcast_1:output:03random_rotation_6/stateful_uniform/Bitcast:output:0Hrandom_rotation_6/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
&random_rotation_6/stateful_uniform/subSub/random_rotation_6/stateful_uniform/max:output:0/random_rotation_6/stateful_uniform/min:output:0*
T0*
_output_shapes
: Í
&random_rotation_6/stateful_uniform/mulMulDrandom_rotation_6/stateful_uniform/StatelessRandomUniformV2:output:0*random_rotation_6/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
"random_rotation_6/stateful_uniformAddV2*random_rotation_6/stateful_uniform/mul:z:0/random_rotation_6/stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
'random_rotation_6/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
%random_rotation_6/rotation_matrix/subSubrandom_rotation_6/Cast_1:y:00random_rotation_6/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 
%random_rotation_6/rotation_matrix/CosCos&random_rotation_6/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
)random_rotation_6/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
'random_rotation_6/rotation_matrix/sub_1Subrandom_rotation_6/Cast_1:y:02random_rotation_6/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: ²
%random_rotation_6/rotation_matrix/mulMul)random_rotation_6/rotation_matrix/Cos:y:0+random_rotation_6/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%random_rotation_6/rotation_matrix/SinSin&random_rotation_6/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
)random_rotation_6/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
'random_rotation_6/rotation_matrix/sub_2Subrandom_rotation_6/Cast:y:02random_rotation_6/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: ´
'random_rotation_6/rotation_matrix/mul_1Mul)random_rotation_6/rotation_matrix/Sin:y:0+random_rotation_6/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
'random_rotation_6/rotation_matrix/sub_3Sub)random_rotation_6/rotation_matrix/mul:z:0+random_rotation_6/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
'random_rotation_6/rotation_matrix/sub_4Sub)random_rotation_6/rotation_matrix/sub:z:0+random_rotation_6/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
+random_rotation_6/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Å
)random_rotation_6/rotation_matrix/truedivRealDiv+random_rotation_6/rotation_matrix/sub_4:z:04random_rotation_6/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
)random_rotation_6/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
'random_rotation_6/rotation_matrix/sub_5Subrandom_rotation_6/Cast:y:02random_rotation_6/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 
'random_rotation_6/rotation_matrix/Sin_1Sin&random_rotation_6/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
)random_rotation_6/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
'random_rotation_6/rotation_matrix/sub_6Subrandom_rotation_6/Cast_1:y:02random_rotation_6/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: ¶
'random_rotation_6/rotation_matrix/mul_2Mul+random_rotation_6/rotation_matrix/Sin_1:y:0+random_rotation_6/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'random_rotation_6/rotation_matrix/Cos_1Cos&random_rotation_6/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
)random_rotation_6/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
'random_rotation_6/rotation_matrix/sub_7Subrandom_rotation_6/Cast:y:02random_rotation_6/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: ¶
'random_rotation_6/rotation_matrix/mul_3Mul+random_rotation_6/rotation_matrix/Cos_1:y:0+random_rotation_6/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
%random_rotation_6/rotation_matrix/addAddV2+random_rotation_6/rotation_matrix/mul_2:z:0+random_rotation_6/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
'random_rotation_6/rotation_matrix/sub_8Sub+random_rotation_6/rotation_matrix/sub_5:z:0)random_rotation_6/rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
-random_rotation_6/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @É
+random_rotation_6/rotation_matrix/truediv_1RealDiv+random_rotation_6/rotation_matrix/sub_8:z:06random_rotation_6/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
'random_rotation_6/rotation_matrix/ShapeShape&random_rotation_6/stateful_uniform:z:0*
T0*
_output_shapes
:
5random_rotation_6/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7random_rotation_6/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7random_rotation_6/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:û
/random_rotation_6/rotation_matrix/strided_sliceStridedSlice0random_rotation_6/rotation_matrix/Shape:output:0>random_rotation_6/rotation_matrix/strided_slice/stack:output:0@random_rotation_6/rotation_matrix/strided_slice/stack_1:output:0@random_rotation_6/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
'random_rotation_6/rotation_matrix/Cos_2Cos&random_rotation_6/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
7random_rotation_6/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        
9random_rotation_6/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
9random_rotation_6/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ®
1random_rotation_6/rotation_matrix/strided_slice_1StridedSlice+random_rotation_6/rotation_matrix/Cos_2:y:0@random_rotation_6/rotation_matrix/strided_slice_1/stack:output:0Brandom_rotation_6/rotation_matrix/strided_slice_1/stack_1:output:0Brandom_rotation_6/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
'random_rotation_6/rotation_matrix/Sin_2Sin&random_rotation_6/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
7random_rotation_6/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        
9random_rotation_6/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
9random_rotation_6/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ®
1random_rotation_6/rotation_matrix/strided_slice_2StridedSlice+random_rotation_6/rotation_matrix/Sin_2:y:0@random_rotation_6/rotation_matrix/strided_slice_2/stack:output:0Brandom_rotation_6/rotation_matrix/strided_slice_2/stack_1:output:0Brandom_rotation_6/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
%random_rotation_6/rotation_matrix/NegNeg:random_rotation_6/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
7random_rotation_6/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        
9random_rotation_6/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
9random_rotation_6/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      °
1random_rotation_6/rotation_matrix/strided_slice_3StridedSlice-random_rotation_6/rotation_matrix/truediv:z:0@random_rotation_6/rotation_matrix/strided_slice_3/stack:output:0Brandom_rotation_6/rotation_matrix/strided_slice_3/stack_1:output:0Brandom_rotation_6/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
'random_rotation_6/rotation_matrix/Sin_3Sin&random_rotation_6/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
7random_rotation_6/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        
9random_rotation_6/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
9random_rotation_6/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ®
1random_rotation_6/rotation_matrix/strided_slice_4StridedSlice+random_rotation_6/rotation_matrix/Sin_3:y:0@random_rotation_6/rotation_matrix/strided_slice_4/stack:output:0Brandom_rotation_6/rotation_matrix/strided_slice_4/stack_1:output:0Brandom_rotation_6/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
'random_rotation_6/rotation_matrix/Cos_3Cos&random_rotation_6/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
7random_rotation_6/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        
9random_rotation_6/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
9random_rotation_6/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ®
1random_rotation_6/rotation_matrix/strided_slice_5StridedSlice+random_rotation_6/rotation_matrix/Cos_3:y:0@random_rotation_6/rotation_matrix/strided_slice_5/stack:output:0Brandom_rotation_6/rotation_matrix/strided_slice_5/stack_1:output:0Brandom_rotation_6/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
7random_rotation_6/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        
9random_rotation_6/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
9random_rotation_6/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ²
1random_rotation_6/rotation_matrix/strided_slice_6StridedSlice/random_rotation_6/rotation_matrix/truediv_1:z:0@random_rotation_6/rotation_matrix/strided_slice_6/stack:output:0Brandom_rotation_6/rotation_matrix/strided_slice_6/stack_1:output:0Brandom_rotation_6/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_maskr
0random_rotation_6/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ù
.random_rotation_6/rotation_matrix/zeros/packedPack8random_rotation_6/rotation_matrix/strided_slice:output:09random_rotation_6/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:r
-random_rotation_6/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ò
'random_rotation_6/rotation_matrix/zerosFill7random_rotation_6/rotation_matrix/zeros/packed:output:06random_rotation_6/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
-random_rotation_6/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :°
(random_rotation_6/rotation_matrix/concatConcatV2:random_rotation_6/rotation_matrix/strided_slice_1:output:0)random_rotation_6/rotation_matrix/Neg:y:0:random_rotation_6/rotation_matrix/strided_slice_3:output:0:random_rotation_6/rotation_matrix/strided_slice_4:output:0:random_rotation_6/rotation_matrix/strided_slice_5:output:0:random_rotation_6/rotation_matrix/strided_slice_6:output:00random_rotation_6/rotation_matrix/zeros:output:06random_rotation_6/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!random_rotation_6/transform/ShapeShape6random_flip_6/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
:y
/random_rotation_6/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1random_rotation_6/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1random_rotation_6/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
)random_rotation_6/transform/strided_sliceStridedSlice*random_rotation_6/transform/Shape:output:08random_rotation_6/transform/strided_slice/stack:output:0:random_rotation_6/transform/strided_slice/stack_1:output:0:random_rotation_6/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:k
&random_rotation_6/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
6random_rotation_6/transform/ImageProjectiveTransformV3ImageProjectiveTransformV36random_flip_6/stateless_random_flip_left_right/add:z:01random_rotation_6/rotation_matrix/concat:output:02random_rotation_6/transform/strided_slice:output:0/random_rotation_6/transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR
random_zoom_6/ShapeShapeKrandom_rotation_6/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:k
!random_zoom_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#random_zoom_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#random_zoom_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
random_zoom_6/strided_sliceStridedSlicerandom_zoom_6/Shape:output:0*random_zoom_6/strided_slice/stack:output:0,random_zoom_6/strided_slice/stack_1:output:0,random_zoom_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
#random_zoom_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿx
%random_zoom_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿo
%random_zoom_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
random_zoom_6/strided_slice_1StridedSlicerandom_zoom_6/Shape:output:0,random_zoom_6/strided_slice_1/stack:output:0.random_zoom_6/strided_slice_1/stack_1:output:0.random_zoom_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
random_zoom_6/CastCast&random_zoom_6/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: v
#random_zoom_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿx
%random_zoom_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿo
%random_zoom_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
random_zoom_6/strided_slice_2StridedSlicerandom_zoom_6/Shape:output:0,random_zoom_6/strided_slice_2/stack:output:0.random_zoom_6/strided_slice_2/stack_1:output:0.random_zoom_6/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
random_zoom_6/Cast_1Cast&random_zoom_6/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: h
&random_zoom_6/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :±
$random_zoom_6/stateful_uniform/shapePack$random_zoom_6/strided_slice:output:0/random_zoom_6/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:g
"random_zoom_6/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?g
"random_zoom_6/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?n
$random_zoom_6/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: ª
#random_zoom_6/stateful_uniform/ProdProd-random_zoom_6/stateful_uniform/shape:output:0-random_zoom_6/stateful_uniform/Const:output:0*
T0*
_output_shapes
: g
%random_zoom_6/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
%random_zoom_6/stateful_uniform/Cast_1Cast,random_zoom_6/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: î
-random_zoom_6/stateful_uniform/RngReadAndSkipRngReadAndSkip6random_zoom_6_stateful_uniform_rngreadandskip_resource.random_zoom_6/stateful_uniform/Cast/x:output:0)random_zoom_6/stateful_uniform/Cast_1:y:0*
_output_shapes
:|
2random_zoom_6/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4random_zoom_6/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4random_zoom_6/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
,random_zoom_6/stateful_uniform/strided_sliceStridedSlice5random_zoom_6/stateful_uniform/RngReadAndSkip:value:0;random_zoom_6/stateful_uniform/strided_slice/stack:output:0=random_zoom_6/stateful_uniform/strided_slice/stack_1:output:0=random_zoom_6/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
&random_zoom_6/stateful_uniform/BitcastBitcast5random_zoom_6/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0~
4random_zoom_6/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
6random_zoom_6/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6random_zoom_6/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:è
.random_zoom_6/stateful_uniform/strided_slice_1StridedSlice5random_zoom_6/stateful_uniform/RngReadAndSkip:value:0=random_zoom_6/stateful_uniform/strided_slice_1/stack:output:0?random_zoom_6/stateful_uniform/strided_slice_1/stack_1:output:0?random_zoom_6/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
(random_zoom_6/stateful_uniform/Bitcast_1Bitcast7random_zoom_6/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0}
;random_zoom_6/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :Õ
7random_zoom_6/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2-random_zoom_6/stateful_uniform/shape:output:01random_zoom_6/stateful_uniform/Bitcast_1:output:0/random_zoom_6/stateful_uniform/Bitcast:output:0Drandom_zoom_6/stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
"random_zoom_6/stateful_uniform/subSub+random_zoom_6/stateful_uniform/max:output:0+random_zoom_6/stateful_uniform/min:output:0*
T0*
_output_shapes
: Å
"random_zoom_6/stateful_uniform/mulMul@random_zoom_6/stateful_uniform/StatelessRandomUniformV2:output:0&random_zoom_6/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
random_zoom_6/stateful_uniformAddV2&random_zoom_6/stateful_uniform/mul:z:0+random_zoom_6/stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
random_zoom_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ç
random_zoom_6/concatConcatV2"random_zoom_6/stateful_uniform:z:0"random_zoom_6/stateful_uniform:z:0"random_zoom_6/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
random_zoom_6/zoom_matrix/ShapeShaperandom_zoom_6/concat:output:0*
T0*
_output_shapes
:w
-random_zoom_6/zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/random_zoom_6/zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/random_zoom_6/zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ó
'random_zoom_6/zoom_matrix/strided_sliceStridedSlice(random_zoom_6/zoom_matrix/Shape:output:06random_zoom_6/zoom_matrix/strided_slice/stack:output:08random_zoom_6/zoom_matrix/strided_slice/stack_1:output:08random_zoom_6/zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
random_zoom_6/zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
random_zoom_6/zoom_matrix/subSubrandom_zoom_6/Cast_1:y:0(random_zoom_6/zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: h
#random_zoom_6/zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
!random_zoom_6/zoom_matrix/truedivRealDiv!random_zoom_6/zoom_matrix/sub:z:0,random_zoom_6/zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 
/random_zoom_6/zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
1random_zoom_6/zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
1random_zoom_6/zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
)random_zoom_6/zoom_matrix/strided_slice_1StridedSlicerandom_zoom_6/concat:output:08random_zoom_6/zoom_matrix/strided_slice_1/stack:output:0:random_zoom_6/zoom_matrix/strided_slice_1/stack_1:output:0:random_zoom_6/zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskf
!random_zoom_6/zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¸
random_zoom_6/zoom_matrix/sub_1Sub*random_zoom_6/zoom_matrix/sub_1/x:output:02random_zoom_6/zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
random_zoom_6/zoom_matrix/mulMul%random_zoom_6/zoom_matrix/truediv:z:0#random_zoom_6/zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
!random_zoom_6/zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
random_zoom_6/zoom_matrix/sub_2Subrandom_zoom_6/Cast:y:0*random_zoom_6/zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: j
%random_zoom_6/zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @¤
#random_zoom_6/zoom_matrix/truediv_1RealDiv#random_zoom_6/zoom_matrix/sub_2:z:0.random_zoom_6/zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 
/random_zoom_6/zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
1random_zoom_6/zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
1random_zoom_6/zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
)random_zoom_6/zoom_matrix/strided_slice_2StridedSlicerandom_zoom_6/concat:output:08random_zoom_6/zoom_matrix/strided_slice_2/stack:output:0:random_zoom_6/zoom_matrix/strided_slice_2/stack_1:output:0:random_zoom_6/zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskf
!random_zoom_6/zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¸
random_zoom_6/zoom_matrix/sub_3Sub*random_zoom_6/zoom_matrix/sub_3/x:output:02random_zoom_6/zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
random_zoom_6/zoom_matrix/mul_1Mul'random_zoom_6/zoom_matrix/truediv_1:z:0#random_zoom_6/zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/random_zoom_6/zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
1random_zoom_6/zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
1random_zoom_6/zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
)random_zoom_6/zoom_matrix/strided_slice_3StridedSlicerandom_zoom_6/concat:output:08random_zoom_6/zoom_matrix/strided_slice_3/stack:output:0:random_zoom_6/zoom_matrix/strided_slice_3/stack_1:output:0:random_zoom_6/zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskj
(random_zoom_6/zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Á
&random_zoom_6/zoom_matrix/zeros/packedPack0random_zoom_6/zoom_matrix/strided_slice:output:01random_zoom_6/zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:j
%random_zoom_6/zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    º
random_zoom_6/zoom_matrix/zerosFill/random_zoom_6/zoom_matrix/zeros/packed:output:0.random_zoom_6/zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
*random_zoom_6/zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Å
(random_zoom_6/zoom_matrix/zeros_1/packedPack0random_zoom_6/zoom_matrix/strided_slice:output:03random_zoom_6/zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:l
'random_zoom_6/zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    À
!random_zoom_6/zoom_matrix/zeros_1Fill1random_zoom_6/zoom_matrix/zeros_1/packed:output:00random_zoom_6/zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
/random_zoom_6/zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           
1random_zoom_6/zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
1random_zoom_6/zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
)random_zoom_6/zoom_matrix/strided_slice_4StridedSlicerandom_zoom_6/concat:output:08random_zoom_6/zoom_matrix/strided_slice_4/stack:output:0:random_zoom_6/zoom_matrix/strided_slice_4/stack_1:output:0:random_zoom_6/zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_maskl
*random_zoom_6/zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Å
(random_zoom_6/zoom_matrix/zeros_2/packedPack0random_zoom_6/zoom_matrix/strided_slice:output:03random_zoom_6/zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:l
'random_zoom_6/zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    À
!random_zoom_6/zoom_matrix/zeros_2Fill1random_zoom_6/zoom_matrix/zeros_2/packed:output:00random_zoom_6/zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
%random_zoom_6/zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :É
 random_zoom_6/zoom_matrix/concatConcatV22random_zoom_6/zoom_matrix/strided_slice_3:output:0(random_zoom_6/zoom_matrix/zeros:output:0!random_zoom_6/zoom_matrix/mul:z:0*random_zoom_6/zoom_matrix/zeros_1:output:02random_zoom_6/zoom_matrix/strided_slice_4:output:0#random_zoom_6/zoom_matrix/mul_1:z:0*random_zoom_6/zoom_matrix/zeros_2:output:0.random_zoom_6/zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
random_zoom_6/transform/ShapeShapeKrandom_rotation_6/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:u
+random_zoom_6/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-random_zoom_6/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-random_zoom_6/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
%random_zoom_6/transform/strided_sliceStridedSlice&random_zoom_6/transform/Shape:output:04random_zoom_6/transform/strided_slice/stack:output:06random_zoom_6/transform/strided_slice/stack_1:output:06random_zoom_6/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:g
"random_zoom_6/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
2random_zoom_6/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Krandom_rotation_6/transform/ImageProjectiveTransformV3:transformed_images:0)random_zoom_6/zoom_matrix/concat:output:0.random_zoom_6/transform/strided_slice:output:0+random_zoom_6/transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR 
IdentityIdentityGrandom_zoom_6/transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´ã
NoOpNoOp7^random_flip_6/stateful_uniform_full_int/RngReadAndSkip2^random_rotation_6/stateful_uniform/RngReadAndSkip.^random_zoom_6/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ´´: : : 2p
6random_flip_6/stateful_uniform_full_int/RngReadAndSkip6random_flip_6/stateful_uniform_full_int/RngReadAndSkip2f
1random_rotation_6/stateful_uniform/RngReadAndSkip1random_rotation_6/stateful_uniform/RngReadAndSkip2^
-random_zoom_6/stateful_uniform/RngReadAndSkip-random_zoom_6/stateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs

ý
D__inference_conv2d_25_layer_call_and_return_conditional_losses_46581

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

d
H__inference_random_flip_6_layer_call_and_return_conditional_losses_44780

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´´:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
®
Î
-__inference_sequential_13_layer_call_fn_45383
input_7!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:
À(
	unknown_8:	
	unknown_9:	

unknown_10:
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_13_layer_call_and_return_conditional_losses_45356o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ´´: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
!
_user_specified_name	input_7
»
L
0__inference_max_pooling2d_24_layer_call_fn_46556

inputs
identityÜ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_45208
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
I
-__inference_random_zoom_6_layer_call_fn_46855

inputs
identityÀ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_random_zoom_6_layer_call_and_return_conditional_losses_44792j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´´:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
¦

÷
C__inference_dense_12_layer_call_and_return_conditional_losses_46622

inputs2
matmul_readvariableop_resource:
À(.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
À(*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ(
 
_user_specified_nameinputs

h
L__inference_random_rotation_6_layer_call_and_return_conditional_losses_44786

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´´:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
©

-__inference_sequential_13_layer_call_fn_45601
input_7
unknown:	
	unknown_0:	
	unknown_1:	#
	unknown_2:
	unknown_3:#
	unknown_4: 
	unknown_5: #
	unknown_6: @
	unknown_7:@#
	unknown_8:@@
	unknown_9:@

unknown_10:
À(

unknown_11:	

unknown_12:	

unknown_13:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_13_layer_call_and_return_conditional_losses_45533o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ´´: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
!
_user_specified_name	input_7

ý
D__inference_conv2d_23_layer_call_and_return_conditional_losses_46521

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿWW *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿWW X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿWW i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿWW w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿYY: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿYY
 
_user_specified_nameinputs
Æ
`
D__inference_flatten_6_layer_call_and_return_conditional_losses_45319

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ(Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ		@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		@
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*µ
serving_default¡
E
input_7:
serving_default_input_7:0ÿÿÿÿÿÿÿÿÿ´´<
dense_130
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ö
®
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
Ñ
layer-0
layer-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_sequential
¥
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
»

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_layer
»

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Bkernel
Cbias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Pkernel
Qbias
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses"
_tf_keras_layer
»

dkernel
ebias
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses"
_tf_keras_layer
»

lkernel
mbias
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses"
_tf_keras_layer
â
titer
	udecay
vlearning_rate
wmomentum
xrho
&rmsõ
'rmsö
4rms÷
5rmsø
Brmsù
Crmsú
Prmsû
Qrmsü
drmsý
ermsþ
lrmsÿ
mrms"
	optimizer
v
&0
'1
42
53
B4
C5
P6
Q7
d8
e9
l10
m11"
trackable_list_wrapper
v
&0
'1
42
53
B4
C5
P6
Q7
d8
e9
l10
m11"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2ÿ
-__inference_sequential_13_layer_call_fn_45383
-__inference_sequential_13_layer_call_fn_45724
-__inference_sequential_13_layer_call_fn_45759
-__inference_sequential_13_layer_call_fn_45601À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
H__inference_sequential_13_layer_call_and_return_conditional_losses_45815
H__inference_sequential_13_layer_call_and_return_conditional_losses_46137
H__inference_sequential_13_layer_call_and_return_conditional_losses_45642
H__inference_sequential_13_layer_call_and_return_conditional_losses_45689À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ËBÈ
 __inference__wrapped_model_44769input_7"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
~serving_default"
signature_map
Â
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2ÿ
-__inference_sequential_12_layer_call_fn_44798
-__inference_sequential_12_layer_call_fn_46173
-__inference_sequential_12_layer_call_fn_46184
-__inference_sequential_12_layer_call_fn_45155À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
H__inference_sequential_12_layer_call_and_return_conditional_losses_46188
H__inference_sequential_12_layer_call_and_return_conditional_losses_46458
H__inference_sequential_12_layer_call_and_return_conditional_losses_45162
H__inference_sequential_12_layer_call_and_return_conditional_losses_45175À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_rescaling_6_layer_call_fn_46463¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_rescaling_6_layer_call_and_return_conditional_losses_46471¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
*:(2conv2d_22/kernel
:2conv2d_22/bias
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_conv2d_22_layer_call_fn_46480¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_22_layer_call_and_return_conditional_losses_46491¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
£non_trainable_variables
¤layers
¥metrics
 ¦layer_regularization_losses
§layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_max_pooling2d_22_layer_call_fn_46496¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_46501¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
*:( 2conv2d_23/kernel
: 2conv2d_23/bias
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_conv2d_23_layer_call_fn_46510¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_23_layer_call_and_return_conditional_losses_46521¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_max_pooling2d_23_layer_call_fn_46526¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_46531¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
*:( @2conv2d_24/kernel
:@2conv2d_24/bias
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
²non_trainable_variables
³layers
´metrics
 µlayer_regularization_losses
¶layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_conv2d_24_layer_call_fn_46540¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_24_layer_call_and_return_conditional_losses_46551¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
·non_trainable_variables
¸layers
¹metrics
 ºlayer_regularization_losses
»layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_max_pooling2d_24_layer_call_fn_46556¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_46561¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
*:(@@2conv2d_25/kernel
:@2conv2d_25/bias
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_conv2d_25_layer_call_fn_46570¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_25_layer_call_and_return_conditional_losses_46581¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ánon_trainable_variables
Âlayers
Ãmetrics
 Älayer_regularization_losses
Ålayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_max_pooling2d_25_layer_call_fn_46586¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_46591¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_flatten_6_layer_call_fn_46596¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_flatten_6_layer_call_and_return_conditional_losses_46602¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
#:!
À(2dense_12/kernel
:2dense_12/bias
.
d0
e1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ënon_trainable_variables
Ìlayers
Ímetrics
 Îlayer_regularization_losses
Ïlayer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_12_layer_call_fn_46611¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_12_layer_call_and_return_conditional_losses_46622¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
": 	2dense_13/kernel
:2dense_13/bias
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_13_layer_call_fn_46631¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_13_layer_call_and_return_conditional_losses_46642¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
 "
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
0
Õ0
Ö1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÊBÇ
#__inference_signature_wrapper_46168input_7"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
·
×non_trainable_variables
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ûlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
/
Ü
_generator"
_generic_user_object
2
-__inference_random_flip_6_layer_call_fn_46647
-__inference_random_flip_6_layer_call_fn_46654´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Î2Ë
H__inference_random_flip_6_layer_call_and_return_conditional_losses_46658
H__inference_random_flip_6_layer_call_and_return_conditional_losses_46716´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ýnon_trainable_variables
Þlayers
ßmetrics
 àlayer_regularization_losses
álayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
/
â
_generator"
_generic_user_object
 2
1__inference_random_rotation_6_layer_call_fn_46721
1__inference_random_rotation_6_layer_call_fn_46728´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
L__inference_random_rotation_6_layer_call_and_return_conditional_losses_46732
L__inference_random_rotation_6_layer_call_and_return_conditional_losses_46850´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ãnon_trainable_variables
älayers
åmetrics
 ælayer_regularization_losses
çlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
/
è
_generator"
_generic_user_object
2
-__inference_random_zoom_6_layer_call_fn_46855
-__inference_random_zoom_6_layer_call_fn_46862´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Î2Ë
H__inference_random_zoom_6_layer_call_and_return_conditional_losses_46866
H__inference_random_zoom_6_layer_call_and_return_conditional_losses_46968´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
R

étotal

êcount
ë	variables
ì	keras_api"
_tf_keras_metric
c

ítotal

îcount
ï
_fn_kwargs
ð	variables
ñ	keras_api"
_tf_keras_metric
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
/
ò
_state_var"
_generic_user_object
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
/
ó
_state_var"
_generic_user_object
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
/
ô
_state_var"
_generic_user_object
:  (2total
:  (2count
0
é0
ê1"
trackable_list_wrapper
.
ë	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
í0
î1"
trackable_list_wrapper
.
ð	variables"
_generic_user_object
": 	2random_flip_6/StateVar
&:$	2random_rotation_6/StateVar
": 	2random_zoom_6/StateVar
4:22RMSprop/conv2d_22/kernel/rms
&:$2RMSprop/conv2d_22/bias/rms
4:2 2RMSprop/conv2d_23/kernel/rms
&:$ 2RMSprop/conv2d_23/bias/rms
4:2 @2RMSprop/conv2d_24/kernel/rms
&:$@2RMSprop/conv2d_24/bias/rms
4:2@@2RMSprop/conv2d_25/kernel/rms
&:$@2RMSprop/conv2d_25/bias/rms
-:+
À(2RMSprop/dense_12/kernel/rms
&:$2RMSprop/dense_12/bias/rms
,:*	2RMSprop/dense_13/kernel/rms
%:#2RMSprop/dense_13/bias/rms£
 __inference__wrapped_model_44769&'45BCPQdelm:¢7
0¢-
+(
input_7ÿÿÿÿÿÿÿÿÿ´´
ª "3ª0
.
dense_13"
dense_13ÿÿÿÿÿÿÿÿÿ¸
D__inference_conv2d_22_layer_call_and_return_conditional_losses_46491p&'9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ´´
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ²²
 
)__inference_conv2d_22_layer_call_fn_46480c&'9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ´´
ª ""ÿÿÿÿÿÿÿÿÿ²²´
D__inference_conv2d_23_layer_call_and_return_conditional_losses_46521l457¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿYY
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿWW 
 
)__inference_conv2d_23_layer_call_fn_46510_457¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿYY
ª " ÿÿÿÿÿÿÿÿÿWW ´
D__inference_conv2d_24_layer_call_and_return_conditional_losses_46551lBC7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ++ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ))@
 
)__inference_conv2d_24_layer_call_fn_46540_BC7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ++ 
ª " ÿÿÿÿÿÿÿÿÿ))@´
D__inference_conv2d_25_layer_call_and_return_conditional_losses_46581lPQ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
)__inference_conv2d_25_layer_call_fn_46570_PQ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@¥
C__inference_dense_12_layer_call_and_return_conditional_losses_46622^de0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÀ(
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
(__inference_dense_12_layer_call_fn_46611Qde0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÀ(
ª "ÿÿÿÿÿÿÿÿÿ¤
C__inference_dense_13_layer_call_and_return_conditional_losses_46642]lm0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
(__inference_dense_13_layer_call_fn_46631Plm0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
D__inference_flatten_6_layer_call_and_return_conditional_losses_46602a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ		@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÀ(
 
)__inference_flatten_6_layer_call_fn_46596T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ		@
ª "ÿÿÿÿÿÿÿÿÿÀ(î
K__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_46501R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_22_layer_call_fn_46496R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_46531R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_23_layer_call_fn_46526R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_46561R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_24_layer_call_fn_46556R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_46591R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_25_layer_call_fn_46586R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¼
H__inference_random_flip_6_layer_call_and_return_conditional_losses_46658p=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ´´
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ´´
 À
H__inference_random_flip_6_layer_call_and_return_conditional_losses_46716tò=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ´´
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ´´
 
-__inference_random_flip_6_layer_call_fn_46647c=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ´´
p 
ª ""ÿÿÿÿÿÿÿÿÿ´´
-__inference_random_flip_6_layer_call_fn_46654gò=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ´´
p
ª ""ÿÿÿÿÿÿÿÿÿ´´À
L__inference_random_rotation_6_layer_call_and_return_conditional_losses_46732p=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ´´
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ´´
 Ä
L__inference_random_rotation_6_layer_call_and_return_conditional_losses_46850tó=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ´´
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ´´
 
1__inference_random_rotation_6_layer_call_fn_46721c=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ´´
p 
ª ""ÿÿÿÿÿÿÿÿÿ´´
1__inference_random_rotation_6_layer_call_fn_46728gó=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ´´
p
ª ""ÿÿÿÿÿÿÿÿÿ´´¼
H__inference_random_zoom_6_layer_call_and_return_conditional_losses_46866p=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ´´
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ´´
 À
H__inference_random_zoom_6_layer_call_and_return_conditional_losses_46968tô=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ´´
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ´´
 
-__inference_random_zoom_6_layer_call_fn_46855c=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ´´
p 
ª ""ÿÿÿÿÿÿÿÿÿ´´
-__inference_random_zoom_6_layer_call_fn_46862gô=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ´´
p
ª ""ÿÿÿÿÿÿÿÿÿ´´¶
F__inference_rescaling_6_layer_call_and_return_conditional_losses_46471l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ´´
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ´´
 
+__inference_rescaling_6_layer_call_fn_46463_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ´´
ª ""ÿÿÿÿÿÿÿÿÿ´´Î
H__inference_sequential_12_layer_call_and_return_conditional_losses_45162N¢K
D¢A
74
random_flip_6_inputÿÿÿÿÿÿÿÿÿ´´
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ´´
 Ö
H__inference_sequential_12_layer_call_and_return_conditional_losses_45175òóôN¢K
D¢A
74
random_flip_6_inputÿÿÿÿÿÿÿÿÿ´´
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ´´
 À
H__inference_sequential_12_layer_call_and_return_conditional_losses_46188tA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ´´
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ´´
 È
H__inference_sequential_12_layer_call_and_return_conditional_losses_46458|òóôA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ´´
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ´´
 ¥
-__inference_sequential_12_layer_call_fn_44798tN¢K
D¢A
74
random_flip_6_inputÿÿÿÿÿÿÿÿÿ´´
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ´´­
-__inference_sequential_12_layer_call_fn_45155|òóôN¢K
D¢A
74
random_flip_6_inputÿÿÿÿÿÿÿÿÿ´´
p

 
ª ""ÿÿÿÿÿÿÿÿÿ´´
-__inference_sequential_12_layer_call_fn_46173gA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ´´
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ´´ 
-__inference_sequential_12_layer_call_fn_46184oòóôA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ´´
p

 
ª ""ÿÿÿÿÿÿÿÿÿ´´Å
H__inference_sequential_13_layer_call_and_return_conditional_losses_45642y&'45BCPQdelmB¢?
8¢5
+(
input_7ÿÿÿÿÿÿÿÿÿ´´
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ë
H__inference_sequential_13_layer_call_and_return_conditional_losses_45689òóô&'45BCPQdelmB¢?
8¢5
+(
input_7ÿÿÿÿÿÿÿÿÿ´´
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ä
H__inference_sequential_13_layer_call_and_return_conditional_losses_45815x&'45BCPQdelmA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ´´
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ê
H__inference_sequential_13_layer_call_and_return_conditional_losses_46137~òóô&'45BCPQdelmA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ´´
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_sequential_13_layer_call_fn_45383l&'45BCPQdelmB¢?
8¢5
+(
input_7ÿÿÿÿÿÿÿÿÿ´´
p 

 
ª "ÿÿÿÿÿÿÿÿÿ£
-__inference_sequential_13_layer_call_fn_45601ròóô&'45BCPQdelmB¢?
8¢5
+(
input_7ÿÿÿÿÿÿÿÿÿ´´
p

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_13_layer_call_fn_45724k&'45BCPQdelmA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ´´
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¢
-__inference_sequential_13_layer_call_fn_45759qòóô&'45BCPQdelmA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ´´
p

 
ª "ÿÿÿÿÿÿÿÿÿ²
#__inference_signature_wrapper_46168&'45BCPQdelmE¢B
¢ 
;ª8
6
input_7+(
input_7ÿÿÿÿÿÿÿÿÿ´´"3ª0
.
dense_13"
dense_13ÿÿÿÿÿÿÿÿÿ
// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package reduce

import (
	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// IDsRowSize retains Query's encoded one-row IDs size. Search Chunk thresholds
// use the PK payload length instead, without protobuf tags or length prefixes.
func IDsRowSize(ids *schemapb.IDs, row int64) (int, error) {
	if ids.GetIdField() == nil {
		return 0, nil
	}
	if row < 0 || row >= int64(typeutil.GetSizeOfIDs(ids)) {
		return 0, merr.WrapErrServiceInternalMsg("Unit row %d is outside field data length %d", row, typeutil.GetSizeOfIDs(ids))
	}
	var field protowire.Number
	var size int
	switch data := ids.GetIdField().(type) {
	case *schemapb.IDs_IntId:
		field, size = 1, protowire.SizeVarint(uint64(data.IntId.Data[row]))
	case *schemapb.IDs_StrId:
		field, size = 2, len(data.StrId.Data[row])
	default:
		return 0, merr.WrapErrServiceInternalMsg("unsupported IDs representation %T", data)
	}
	arraySize := protowire.SizeTag(1) + protowire.SizeBytes(size)
	return protowire.SizeTag(field) + protowire.SizeBytes(arraySize), nil
}

func scalarPayloadSize[T any](values []T, row int64, null bool, size func(T) int) (int, error) {
	var value T
	if row >= 0 && row < int64(len(values)) {
		value = values[row]
	} else if !null {
		return 0, merr.WrapErrServiceInternalMsg("Unit row %d is outside field data length %d", row, len(values))
	}
	return size(value), nil
}

// FieldDataRowSize counts column payload, not its protobuf envelope. row is
// the logical hit; dataRow is its payload slot (-1 for a NULL group key).
func FieldDataRowSize(field *schemapb.FieldData, row, dataRow int64) (int, error) {
	if field == nil || row < 0 {
		return 0, merr.WrapErrServiceInternalMsg("invalid Unit field or row %d", row)
	}
	valid := typeutil.GetFieldDataValidData(field)
	if len(valid) > 0 && row >= int64(len(valid)) {
		return 0, merr.WrapErrServiceInternalMsg("Unit row %d is outside validity length %d", row, len(valid))
	}
	null := dataRow < 0 || (len(valid) > 0 && !valid[row])
	switch data := field.GetField().(type) {
	case *schemapb.FieldData_Scalars:
		return scalarFieldRowSize(data.Scalars, dataRow, null)
	case *schemapb.FieldData_Vectors:
		if null {
			return 0, nil
		}
		return vectorFieldRowSize(data.Vectors, dataRow)
	case *schemapb.FieldData_StructArrays:
		size := 0
		for _, child := range data.StructArrays.GetFields() {
			n, err := FieldDataRowSize(child, row, row)
			if err != nil {
				return 0, err
			}
			size += n
		}
		return size, nil
	default:
		return 0, merr.WrapErrServiceInternalMsg("unsupported Unit field representation %T", data)
	}
}

func scalarFieldRowSize(field *schemapb.ScalarField, row int64, null bool) (int, error) {
	// Fixed-width NULL scalars still occupy their zero-filled slot in the
	// result. In particular, all three IntArray-backed types use four bytes.
	int32Size := func(int32) int { return 4 }
	int64Size := func(int64) int { return 8 }
	stringSize := func(v string) int { return len(v) }
	bytesSize := func(v []byte) int { return len(v) }
	switch data := field.GetData().(type) {
	case *schemapb.ScalarField_BoolData:
		return scalarPayloadSize(data.BoolData.GetData(), row, null, func(bool) int { return 1 })
	case *schemapb.ScalarField_IntData:
		return scalarPayloadSize(data.IntData.GetData(), row, null, int32Size)
	case *schemapb.ScalarField_LongData:
		return scalarPayloadSize(data.LongData.GetData(), row, null, int64Size)
	case *schemapb.ScalarField_FloatData:
		return scalarPayloadSize(data.FloatData.GetData(), row, null, func(float32) int { return 4 })
	case *schemapb.ScalarField_DoubleData:
		return scalarPayloadSize(data.DoubleData.GetData(), row, null, func(float64) int { return 8 })
	case *schemapb.ScalarField_StringData:
		return scalarPayloadSize(data.StringData.GetData(), row, null, stringSize)
	case *schemapb.ScalarField_BytesData:
		return scalarPayloadSize(data.BytesData.GetData(), row, null, bytesSize)
	case *schemapb.ScalarField_ArrayData:
		return scalarPayloadSize(data.ArrayData.GetData(), row, null, func(v *schemapb.ScalarField) int { return proto.Size(v) })
	case *schemapb.ScalarField_JsonData:
		return scalarPayloadSize(data.JsonData.GetData(), row, null, bytesSize)
	case *schemapb.ScalarField_GeometryData:
		return scalarPayloadSize(data.GeometryData.GetData(), row, null, bytesSize)
	case *schemapb.ScalarField_TimestamptzData:
		return scalarPayloadSize(data.TimestamptzData.GetData(), row, null, int64Size)
	case *schemapb.ScalarField_GeometryWktData:
		return scalarPayloadSize(data.GeometryWktData.GetData(), row, null, stringSize)
	case *schemapb.ScalarField_MolData:
		return scalarPayloadSize(data.MolData.GetData(), row, null, bytesSize)
	case *schemapb.ScalarField_MolSmilesData:
		return scalarPayloadSize(data.MolSmilesData.GetData(), row, null, stringSize)
	case *schemapb.ScalarField_DateData:
		return scalarPayloadSize(data.DateData.GetData(), row, null, int32Size)
	case *schemapb.ScalarField_TimeData:
		return scalarPayloadSize(data.TimeData.GetData(), row, null, int64Size)
	case nil:
		return 0, nil
	default:
		return 0, merr.WrapErrServiceInternalMsg("unsupported Unit scalar representation %T", data)
	}
}

func vectorFieldRowSize(field *schemapb.VectorField, row int64) (int, error) {
	dim := field.GetDim()
	var width, length int64
	switch data := field.GetData().(type) {
	case *schemapb.VectorField_FloatVector:
		width, length = dim*4, int64(len(data.FloatVector.GetData()))*4
	case *schemapb.VectorField_BinaryVector:
		width, length = dim/8, int64(len(data.BinaryVector))
	case *schemapb.VectorField_Float16Vector:
		width, length = dim*2, int64(len(data.Float16Vector))
	case *schemapb.VectorField_Bfloat16Vector:
		width, length = dim*2, int64(len(data.Bfloat16Vector))
	case *schemapb.VectorField_Int8Vector:
		width, length = dim, int64(len(data.Int8Vector))
	case *schemapb.VectorField_SparseFloatVector:
		return scalarPayloadSize(data.SparseFloatVector.GetContents(), row, false, func(v []byte) int { return len(v) })
	case *schemapb.VectorField_VectorArray:
		return scalarPayloadSize(data.VectorArray.GetData(), row, false, func(v *schemapb.VectorField) int { return proto.Size(v) })
	case nil:
		return 0, nil
	default:
		return 0, merr.WrapErrServiceInternalMsg("unsupported Unit vector representation %T", data)
	}
	if dim < 0 || row < 0 || width < 0 || width > length || row >= length/max(1, width) {
		return 0, merr.WrapErrServiceInternalMsg("Unit vector row %d exceeds payload length %d (dim %d)", row, length, dim)
	}
	return int(width), nil
}

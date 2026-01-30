"use strict";

Process.prototype.doCallSiu = function (rest, body, headers){
    var url0 = location.protocol + "//" + location.host + "/siu-star/services/api" + rest;
    if (!headers) {
        headers = "Content-Type: application/json;charset=utf-8"
    }
    var result = null;
    this.stepFrameCount = 0;
    while (result == null && (this.currentTime - this.lastYield) < this.timeout){
        result = this.reportURL_extended(url0, body, headers);
        this.stepFrameCount += 1;
        if (this.stepFrameCount > 100) {
            this.currentTime = Date.now();
            this.stepFrameCount = 0;
        }
        this.popContext();
    }
    if (result == null) {
        return new SnapObject({"error":"connection time out"});
    }
    return result;
}

Process.prototype.reportSiuNauTirIds = function (nauId) {
    var rest = "/nau/" + nauId + "/tirs";
    var tirs = this.doCallSiu(rest, null);
    if (tirs instanceof List){
        tirs = tirs.contents;
    }
    var tir_ids = {};
    for (var j = 0; j < tirs.length; ++j){
        var tir = tirs[j];
        if (tir instanceof SnapObject){
            tir = tir.data;
        }
        tir_ids[tir["name"]] = tir["id"];
    }
    return new SnapObject(tir_ids);
}

Process.prototype.reportSiuTirMetas = function (tirId) {
    var rest = "/tir/" + tirId + "/metas";
    var body = {
        "depth":1,
        "withBaseMetas":false,
        "withDictChilds":true,
        "withDictChildsAsObject":true
    }
    var metas = this.doCallSiu(rest, JSON.stringify(body));
    if (metas instanceof List){
        metas = metas.contents;
    }
    var meta_ids = {};
    for (var j = 0; j < metas.length; ++j){
        var meta = metas[j];
        if (meta instanceof SnapObject){
            meta = meta.data;
        }
        meta_ids[meta["name"]] = meta;
    }
    return new SnapObject(meta_ids);
}

Process.prototype.reportSiuCreateFolder = function (folderName, parentFolderId) {
    var rest = "/folder/" + parentFolderId + "/childs/find"
    var body = '{"name":"' + folderName + '"}';
    var old_folder = this.doCallSiu(rest, body);

    var folder_data = old_folder;
    if (folder_data instanceof SnapObject) {
        folder_data = folder_data.data;
    }
    if (!folder_data.hasOwnProperty("error") || folder_data["error"].indexOf("not found") == -1) {
        return old_folder;
    } else {
        var rest = "/folder/" + parentFolderId + "/childs"
        var new_folder = this.doCallSiu(rest, body);
        return new_folder;
    }
}

Process.prototype.reportSiuCreateMetaValue = function (meta, value) {
    if (meta instanceof SnapObject){
        meta = meta.data;
    }
    if (value instanceof SnapDate){
        value = value.data.toString("dd.MM.yyyy")
    } else
    if (value instanceof SnapDateTime){
        value = value.data.toString("dd.MM.yyyy HH:mm")
    }
    var tim = {};
    tim["timId"] = meta["id"];
    tim["name"] = meta["name"];
    tim["typeCode"] = (meta["typeMeta"].data)["id"];
    tim["value"] = value;
    return new SnapObject(tim)
}


Process.prototype.reportSiuCreateIr = function (irvName, parentFolderId, nauId,
                                                description, comment, metadata, fileName) {
    // Сначал проверим наличие версии IR с указанным имененм
    var rest = "/folder/" + parentFolderId + "/irvs/find"
    var irv = {
        "name": irvName,
        "withMetaData": false
    }
    var irvs = this.doCallSiu(rest, JSON.stringify(irv));
    if (irvs instanceof List) {
        irvs = irvs.contents;
    }
    for (var j = 0; j < irvs.length; ++j) {
        var the_irv = irvs[j];
        if (the_irv instanceof SnapObject) {
            the_irv = the_irv.data
        }
        if (the_irv["name"] == irvName) {
            irv = the_irv;
            break;
        }
    }
    var metadataXML = null;
    if (metadata != null) {
        metadataXML = this.reportXML(metadata);
    }
    irv["xmlMetaDataString"] = metadataXML
    if (description == null || description.length == 0) {
        description = irvName;
    }
    irv["description"] = description;
    if (comment != null && comment.length > 0) {
        irv["comment"] = comment;
    }
    irv["nauId"] = nauId;
    if (irv["ir"] != null) {
        irv["ioId"] = (irv["ir"].data)["id"]
    }
    if (fileName != null) {
        if (fileName instanceof List){
            irv["fileName"] = fileName.contents.join("&comma;");
        } else {
            irv["fileName"] = fileName;
        }
        irv["fileNameSeparator"] = "&comma;";
    }

    var rest = "/folder/" + parentFolderId + "/irvs"
    var new_irv = this.doCallSiu(rest, JSON.stringify(irv));
    return new_irv;
}

Process.prototype.reportSiuCreateIrMetadataDict = function (tirId, metaValues) {
    var operationGetTypeData = {}
    operationGetTypeData["typeIOId"] = tirId;
    operationGetTypeData["typeIOMetaList"] = {"typeIOMeta":metaValues};
    var metaDataObj = {};
    metaDataObj["operationGetTypeData"] = operationGetTypeData;
    return new SnapObject(metaDataObj)
}

Process.prototype.reportSiuNauFolders = function (nauId) {
    var rest = "/nau/" + nauId + "/folders"
    var folders = this.doCallSiu(rest, null);
    return folders
}

Process.prototype.reportSiuNauChilds = function (folderId) {
    var rest = "/folder/" + folderId + "/childs"
    var folders = this.doCallSiu(rest, null);
    return folders
}

Process.prototype.reportSiuFolderIrvs = function (folderId) {
    var rest = "/folder/" + folderId + "/irvs"
    var irvs = this.doCallSiu(rest, null);
    return irvs
}

Process.prototype.reportSiuIrv = function (irvId) {
    var rest = "/irv/" + irvId
    var body = {
        attrKey:"name",
        withMeta:true,
        withBaseMetas:true,
        withSemantic:true,
        withFiles:true,
        planeValues:false,
        withDictChilds:false,
        withDictChildsAsObject:false,
        depth:0,
        dictSortOrder:"name"
    }
    var irv = this.doCallSiu(rest, JSON.stringify(body));
    return irv
}

Process.prototype.reportSiuIrvFiles = function (irv) {
    if (irv instanceof String) {
      var rest = "/irv/" + irv + "/files"
      var files = this.doCallSiu(rest, null);
      return files
    } else
    if (!(irv instanceof SnapObject)) {
        throw new Error("Объект на входе не является ни идентификатором, ни словарем")
    }
    if (!("attrMap" in irv.data)) {
        if (typeof irv.data.error == "undefined") {
            throw new Error("Объект на входе не является ИР")
        } else {
            throw irv.data.error
        }
    }
    var attrMap = irv.data["attrMap"];
    if (attrMap == null){
        throw new Error("ИР на входе не имеет атрибутов")
    }
    if (!("Файлы" in attrMap.data)) {
        var rest = "/irv/" + irv.data["id"] + "/files"
        var files = this.doCallSiu(rest, null);
        return files    }
    var files = (attrMap.data["Файлы"]).data["value"]
    if (!(files) instanceof List){
        throw new Error("Список файлов имеет не правильный формат")
    }
    return files
}

Process.prototype.reportSiuPostIrvFileContent = function (file, body) {
    if (!(file instanceof SnapObject)) {
        throw new Error("Объект на входе не является словарем")
    }
    if (!("irvfId" in file.data)) {
        throw new Error("Объект на входе не писывает файл")
    }
    var irvfId = file.data["irvfId"]
    var fileName = file.data["name"]

    var headers = ""

    if (body instanceof String){
        headers += "content-type:plain/text;charset=utf-8"
    } else {
//        headers += "content-type:application/octet-stream;charset=utf-8"
        headers += "content-type:application/octet-stream"
    }

    var crc
    if (typeof(body) == "string"){
       crc = window.SparkMD5.hash(body)
    } else {
        var spark = new window.SparkMD5.ArrayBuffer()
        spark.append(body)
        crc = spark.end()
    }
    var rest = "/file/" + irvfId + "/write?fileName=" + encodeURIComponent(fileName) + "&crc=" + crc
    var result = this.doCallSiu(rest, body, headers);
    return result
}

Process.prototype.reportSiuGetIrvFileContent = function (file) {
    if (!(file instanceof SnapObject)) {
        throw new Error("Объект на входе не является словарем")
    }
    if (!("irvfId" in file.data)) {
        throw new Error("Объект на входе не писывает файл")
    }
    var irvfId = file.data["irvfId"]
    var rest = "/file/" + irvfId + "/read"
    var filesContent = this.doCallSiu(rest, null);
    return filesContent
}

Process.prototype.reportSiuGetIrvFileStatus = function (file) {
    if (!(file instanceof SnapObject)) {
        throw new Error("Объект на входе не является словарем")
    }
    if (!("irvfId" in file.data)) {
        throw new Error("Объект на входе не писывает файл")
    }
    var irvfId = file.data["irvfId"]
    var rest = "/file/" + irvfId + "/status"
    var result = this.doCallSiu(rest, null);
    return result.data["uploadStatus"]
}

Process.prototype.reportSiuPostIrvFileContentFromExternalURL = function (file, url) {
    // var headers = null;
    // var callback = function(status, response){
    //     this.reportSiuPostIrvFileContent(file, response);
    // }
    // return this.reportURL_extended_async(url, null, headers, true, callback)
    return SnapSiuExternalFileUploader.uploadExternalFile(this, file, url)
 }


/**********************************************************************/

Process.prototype.reportSiuDictTreeElement = function(rootDict, input) {
    // rootDict не используется - нужен только на этапе разработки алгоритма
    var varFrame = this.context.variables;
    var dictId;
    if (input instanceof IdObject){
        dictId = input.data.id;
    } else {
        dictId = input.id;
    }
    var globalObjects = varFrame.getGlobalObjects();
    var dictionary = globalObjects[dictId];
    if (dictionary instanceof IdObject) {
        return dictionary;
    } else {
        return new IdObject(dictionary);
    }
}

Process.prototype.reportSiuDictName = function(dict){
    if (! dict instanceof IdObject) {
        throw new Exception("Объект на входе на является узлом классификатора СИУ");
    }
    return dict.data.name
}

Process.prototype.reportSiuDictName = function(dict){
    if (! dict instanceof IdObject) {
        throw new Exception("Объект на входе на является узлом классификатора СИУ");
    }
    return dict.data.name
}

Process.prototype.reportSiuDictId = function(dict){
    if (! dict instanceof IdObject) {
        throw new Exception("Объект на входе на является узлом классификатора СИУ");
    }
    return dict.data.id
}

Process.prototype.reportSiuDictCode = function(dict){
    if (! dict instanceof IdObject) {
        throw new Exception("Объект на входе на является узлом классификатора СИУ");
    }
    if (typeof dict.data.code == "undefined") {
        return null
    } else {
        return dict.data.code
    }
}

Process.prototype.reportSiuDictValue = function(dict){
    if (! dict instanceof IdObject) {
        throw new Exception("Объект на входе на является узлом классификатора СИУ");
    }
    if (typeof dict.data.paramValue == "undefined") {
        return null
    } else {
        return dict.data.paramValue
    }
}

Process.prototype.setSiuDictValue = function(dict, value){
    if (! dict instanceof IdObject) {
        throw new Exception("Объект на входе на является узлом классификатора СИУ");
    }
    dict.data.paramValue = value
}


Process.prototype.reportSiuDictParentId = function(dict){
    if (! dict instanceof IdObject) {
        throw new Exception("Объект на входе на является узлом классификатора СИУ");
    }
    return dict.data.parentId
}


Process.prototype.reportSiuDictParentName = function(dict){
    if (! dict instanceof IdObject) {
        throw new Exception("Объект на входе на является узлом классификатора СИУ");
    }
    var varFrame = this.context.variables;
    var globalObjects = varFrame.getGlobalObjects();
    var dictionary = globalObjects[dict.data.parentId];
    return dictionary.name
}
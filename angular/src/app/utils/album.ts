import { environment } from "src/environments/environment";

export class albumMeta{
    images:number=0;
    name:string="";
    constructor(name:string, images:number){
        this.name = name;
        this.images = images;
    }
    url(){
        return environment.api+"/picture/"+this.name;
    }

}